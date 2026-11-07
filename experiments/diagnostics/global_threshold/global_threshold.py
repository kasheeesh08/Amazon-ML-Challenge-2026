#!/usr/bin/env python3
"""Global threshold sensitivity (FINAL diagnostic; nothing in production is modified).

Test side (unlabelled): the exact final rule (probability >= t, one-S1-per-target) is re-applied to the SAVED final-model
probabilities (experiments/final/test_prob.npy over experiments/test_scale/cand_test.npy) for t in THRESHOLDS; t = 0.69
must reproduce the submitted predictions (experiments/final/test_pred_codes.npy) exactly. Removed predictions are
profiled (country, probability band, generic names, house-number conflicts, t_rank / t_ncand).
Labelled side: the same thresholds applied to the SAVED Step 3-model probabilities of the labelled environments
V (normal validation), T (test-ratio), S (full-pool stress) — the final model itself cannot be evaluated on labels it was
trained on. No retraining, no inference, no new submission.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.features import FeatureStore  # noqa: E402
from ber.normalize import h64  # noqa: E402
from test_scale_check import sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
TS, FD, TD, MD, SD = (EXP / "test_scale", EXP / "final", EXP / "diagnostics" / "trank", EXP / "matching",
                      EXP / "test_scale" / "stress")
BASE = 0.69
THRESHOLDS = [0.70, 0.72, 0.75, 0.78, 0.80]
BANDS = [(0.69, 0.70), (0.70, 0.72), (0.72, 0.75), (0.75, 0.78), (0.78, 0.80)]
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
C = MF.COL


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def tkey(codes):
    _, s, t = B.decode(codes)
    return (s.astype(np.int64) << 32) | t


def main():
    t0 = time.time()
    hb = {f: sha256(ROOT / f) for f in PROTECTED}
    hb_final = hash_tree(FD)
    res = {"test": {}, "labelled": {}}

    # ------------------------------------------------------------------ test (unlabelled, final model)
    S1 = FeatureStore(TS / "features_test", "test_s1")
    pools = {2: FeatureStore(TS / "features_test", "test_s2"), 3: FeatureStore(TS / "features_test", "test_s3")}
    n1 = S1.n
    c1 = np.asarray(S1.col("country"))
    names = np.asarray(S1.col("name_n2"))
    freq = np.zeros(n1, dtype=np.int64)
    for h in CTRY.values():
        m = c1 == h
        _, inv, cnt = np.unique(names[m], return_inverse=True, return_counts=True)
        freq[m] = cnt[inv]
    cand = np.load(TS / "cand_test.npy")
    pf = np.load(FD / "test_prob.npy")
    base_keep = M.select(cand, pf, BASE, True)
    base_idx = np.flatnonzero(base_keep)
    submitted = np.load(FD / "test_pred_codes.npy")
    res["test"]["baseline_reproduces_submission"] = bool(np.array_equal(cand[base_idx], submitted))
    assert res["test"]["baseline_reproduces_submission"]
    bp = pf[base_idx]
    bq = cand[base_idx] >> B.Q_SHIFT
    bc = c1[bq]
    tr = np.load(TS / "t_rank_test.npy", mmap_mode="r")[base_idx]
    tn = np.load(TS / "t_ncand_test.npy", mmap_mode="r")[base_idx]
    idf_n, idf_a = (MF.IdfTable.load(TS / f"idf_test_{c}.npz") for c in ("name", "addr"))
    hc = np.empty(len(base_idx), dtype=bool)
    legal = np.empty(len(base_idx), dtype=bool)
    for a in range(0, len(base_idx), 1_000_000):
        X = MF.pair_features(S1, np.arange(n1), pools, cand[base_idx[a:a + 1_000_000]], idf_n, idf_a)
        sl = slice(a, a + X.shape[0])
        hc[sl] = (X[:, C["house_both"]] == 1) & (X[:, C["eq_house"]] == 0)
        legal[sl] = (X[:, C["eq_name_n2"]] == 1) & (X[:, C["eq_name_n1"]] == 0)
    gen = freq[bq] >= 10
    log("  test baseline profiled")

    def s1_stats(keep_mask, mask_s1):
        npred = np.bincount(bq[keep_mask], minlength=n1)[mask_s1]
        return {"s1": int(mask_s1.sum()), "predictions": int(npred.sum()), "pred_per_s1": float(npred.mean()),
                "s1_no_prediction": int((npred == 0).sum()), "s1_no_prediction_rate": float((npred == 0).mean())}

    allkeep = np.ones(len(base_idx), dtype=bool)
    res["test"]["0.69"] = {"all": s1_stats(allkeep, np.ones(n1, dtype=bool)),
                           "by_country": {c: s1_stats(allkeep & (bc == h), c1 == h) for c, h in CTRY.items()},
                           "share_house_conflict": float(hc.mean()), "share_generic10": float(gen.mean())}
    for th in THRESHOLDS:
        re_keep = M.select(cand, pf, th, True)
        filt = bp >= th
        same = bool(np.array_equal(np.flatnonzero(re_keep), base_idx[filt]))
        rm = ~filt
        r = {"rule_reapplied_equals_filter_of_submission": same,
             "one_per_target_satisfied": bool(len(np.unique(tkey(cand[base_idx[filt]]))) == int(filt.sum())),
             "all": s1_stats(filt, np.ones(n1, dtype=bool)),
             "removed": int(rm.sum()), "removed_pct": float(rm.mean()),
             "removed_by_band": {f"[{a:.2f},{b:.2f})": int(((bp >= a) & (bp < b)).sum()) for a, b in BANDS if a < th},
             "by_country": {}, "removed_profile": {
                 "t_rank0": float((tr[rm] == 0).mean()), "t_rank0_uncontested": float(((tr[rm] == 0) & (tn[rm] == 1)).mean()),
                 "t_ncand_median": float(np.median(tn[rm])), "house_conflict": float(hc[rm].mean()),
                 "legal_form_only": float(legal[rm].mean()), "generic10_s1": float(gen[rm].mean())},
             "removal_rate_house_conflict_vs_other": [float(rm[hc].mean()), float(rm[~hc].mean())],
             "removal_rate_generic10_vs_unique": [float(rm[gen].mean()), float(rm[freq[bq] == 1].mean())]}
        for c, h in CTRY.items():
            mc = bc == h
            st = s1_stats(filt & mc, c1 == h)
            st["removed"] = int((rm & mc).sum())
            st["removed_pct_of_country"] = float((rm & mc).sum() / mc.sum())
            npg = np.bincount(bq[filt & mc], minlength=n1)
            gm = (c1 == h) & (freq >= 10)
            um = (c1 == h) & (freq == 1)
            st["no_prediction_rate_generic10_vs_unique"] = [float((npg[gm] == 0).mean()), float((npg[um] == 0).mean())]
            r["by_country"][c] = st
        res["test"][str(th)] = r
        log(f"  test t={th}: removed {r['removed']:,} ({100 * r['removed_pct']:.2f}%), no-pred S1 {r['all']['s1_no_prediction']:,}; "
            f"rule==filter {same}")
    del cand, pf

    # ------------------------------------------------------------------ labelled environments (Step 3 model)
    from blocking_experiment import Context
    ctx = Context(EXP / "blocking" / "features", EXP / "validation_split", EXP / "_eda_cache")
    qv = np.flatnonzero(ctx.fold[1] == 1)
    nq = len(qv)
    cv = np.asarray(ctx.s[1].col("country"))[qv]
    true = ctx.truth(qv)
    ntrue = np.bincount(true >> B.Q_SHIFT, minlength=nq)
    is_t = {k: np.zeros(ctx.s[k].n, dtype=bool) for k in (2, 3)}
    for k in (2, 3):
        is_t[k][ctx.pair_trow[ctx.pair_src == k]] = True
    for env, cp in (("V", MD / "cand_val.npy"), ("T", TD / "cand_T.npy"), ("S", SD / "cand_stress.npy")):
        codes = np.load(cp)
        p = np.load(TD / f"p_{env}_A.npy")
        q, s, t = B.decode(codes)
        y = np.isin(codes, true, assume_unique=True)
        orph = np.zeros(len(codes), dtype=bool)
        for k in (2, 3):
            mk = s == k
            orph[mk] = (ctx.fold[k][t[mk]] == 0) & is_t[k][t[mk]]
        base = M.select(codes, p, BASE, True)
        res["labelled"][env] = {}
        for th in [BASE] + THRESHOLDS:
            keep = M.select(codes, p, th, True)
            _, parts = M.entity_f05(codes[keep], true, nq, return_parts=True)
            F, npred = parts["F"], parts["npred"]
            tp = int((keep & y).sum())
            rmv = base & ~keep
            single = ntrue == 0
            r = {"macro_f05": float(F.mean()), "precision": tp / int(keep.sum()), "recall": tp / len(true),
                 "fp": int((keep & ~y).sum()), "fn": int(len(true) - tp), "predictions": int(keep.sum()),
                 "singleton_f05": float(F[single].mean()), "singleton_false_merge_rate": float((npred[single] > 0).mean()),
                 "fp_on_orphan_targets": int((keep & ~y & orph).sum()),
                 "removed_vs_0.69": int(rmv.sum()), "removed_pct": float(rmv.sum() / base.sum()),
                 "removed_true_share": float(y[rmv].mean()) if rmv.any() else None,
                 "by_country_f05": {c: float(F[cv == CTRY[c]].mean()) for c in ("US", "India")}}
            res["labelled"][env][str(th)] = r
        b = res["labelled"][env][str(BASE)]
        for th in THRESHOLDS:
            r = res["labelled"][env][str(th)]
            r["delta_f05"] = r["macro_f05"] - b["macro_f05"]
        log(f"  labelled {env}: " + ", ".join(f"{th}:{res['labelled'][env][str(th)]['macro_f05']:.5f}" for th in [BASE] + THRESHOLDS))

    # ------------------------------------------------------------------ scenario (clearly labelled; not a prediction)
    sc = {}
    for th in THRESHOLDS:
        r_test = res["test"][str(th)]["removed_pct"]
        row = {"test_removed_pct": r_test}
        for env in ("V", "T", "S"):
            L = res["labelled"][env][str(th)]
            row[f"if_test_behaves_like_{env}"] = (L["delta_f05"] * r_test / L["removed_pct"]) if L["removed_pct"] else None
            row[f"{env}_removed_true_share"] = L["removed_true_share"]
        sc[str(th)] = row
    res["scenario_delta_overall_f05"] = {
        "method": "labelled delta-F0.5 per share of predictions removed, scaled by the share actually removed on test "
                  "(linear, crude); S is an orphan-heavy extreme, not a test analogue",
        "rows": sc}

    ha = {f: sha256(ROOT / f) for f in PROTECTED}
    ha_final = hash_tree(FD)
    res["integrity"] = {"protected_unchanged": ha == hb, "n_protected": len(hb),
                        "experiments_final_unchanged": ha_final == hb_final, "n_final_files": len(hb_final),
                        "output/matching_results.tsv": ha["output/matching_results.tsv"],
                        "output/candidate_pairs.tsv": ha["output/candidate_pairs.tsv"],
                        "camelcase_submission.zip": ha["camelcase_submission.zip"],
                        "experiments/final/model_final.npz": ha_final.get("experiments/final/model_final.npz")}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  integrity {res['integrity']}; {res['seconds']}s")


if __name__ == "__main__":
    main()
