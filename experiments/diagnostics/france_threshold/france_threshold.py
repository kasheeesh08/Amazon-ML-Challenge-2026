#!/usr/bin/env python3
"""France-only threshold sensitivity (READ-ONLY diagnostic; no labels exist for France).

Starting point = the EXACT submitted predictions (experiments/final/test_pred_codes.npy, verified pair-for-pair
against output/matching_results.tsv). For each France threshold t in {0.80, 0.82, 0.85, 0.88} the simulated set is
    baseline predictions of US / India S1  (unchanged)
  + baseline predictions of France S1 with final-model probability >= t
i.e. a pure FILTER of the submitted set: the one-per-target rule, candidates and probabilities are unchanged, and a
subset of a one-per-target-valid set stays valid (re-checked). Nothing is written to output/ and no submission file is
produced; all artifacts go to experiments/diagnostics/france_threshold/.

Labelled comparison (separate, no France score): the same filter applied to validation predictions in the already
existing environments V (normal) and T (test-ratio) — the current model on US / India, and the LOCO models in the
over-merging direction (India-only model on US) and under-merging direction (US-only model on India).
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
from test_scale_check import jdump, sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
TS = ROOT / "experiments" / "test_scale"
FD = ROOT / "experiments" / "final"
TD = ROOT / "experiments" / "diagnostics" / "trank"
LD = ROOT / "experiments" / "diagnostics" / "leave_one_country_out"
MD = ROOT / "experiments" / "matching"
BASE_T = 0.69
THRESHOLDS = [0.80, 0.82, 0.85, 0.88]
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
C = MF.COL
PROB_EDGES = [0.69, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88]
RANK_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 9)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tkey(codes):
    _, s, t = B.decode(codes)
    return (s.astype(np.int64) << 32) | t


def rank_dist(r):
    return {(f"{a}" if a == b else (f"{a}+" if b > 10 ** 8 else f"{a}-{b}")): int(((r >= a) & (r <= b)).sum())
            for a, b in RANK_BUCKETS}


def prob_dist(p, hi):
    edges = [e for e in PROB_EDGES if e < hi] + [hi]
    return {f"[{a:.2f},{b:.2f})": int(((p >= a) & (p < b)).sum()) for a, b in zip(edges[:-1], edges[1:])}


def main():
    t0 = time.time()
    hashes_before = {f: sha256(ROOT / f) for f in PROTECTED}
    jdump(OUT / "hashes_before.json", hashes_before)

    # ------------------------------------------------------------------ inputs
    S1 = FeatureStore(TS / "features_test", "test_s1")
    pools = {2: FeatureStore(TS / "features_test", "test_s2"), 3: FeatureStore(TS / "features_test", "test_s3")}
    n1 = S1.n
    c1 = np.asarray(S1.col("country"))
    ids = {1: np.asarray(S1.col("id")), 2: np.asarray(pools[2].col("id")), 3: np.asarray(pools[3].col("id"))}
    cand = np.load(TS / "cand_test.npy")
    p_all = np.load(FD / "test_prob.npy", mmap_mode="r")
    tr_all = np.load(TS / "t_rank_test.npy", mmap_mode="r")
    tn_all = np.load(TS / "t_ncand_test.npy", mmap_mode="r")
    pred = np.load(FD / "test_pred_codes.npy")
    idx = np.searchsorted(cand, pred)
    assert (cand[idx] == pred).all()
    p = np.asarray(p_all[idx])
    tr = np.asarray(tr_all[idx])
    tn = np.asarray(tn_all[idx])
    q, src, t = B.decode(pred)
    ctry = c1[q]
    fr = ctry == CTRY["France"]

    # ------------------------------------------------------------------ baseline checks
    checks = {}
    # 1) saved codes == submitted TSV, pair for pair
    s1_num, tsrc, tnum = [], [], []
    with open(ROOT / "output" / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        rows = 0
        for line in f:
            rows += 1
            a, b = line.rstrip("\n").split("\t")
            if b:
                n = int(a[3:])
                for x in b.split(","):
                    s1_num.append(n)
                    tsrc.append(int(x[1]))
                    tnum.append(int(x[3:]))
    file_keys = np.sort((np.array(s1_num, np.int64) << 34) | (np.array(tsrc, np.int64) << 32) | np.array(tnum, np.int64))
    tid = np.where(src == 2, ids[2][np.minimum(t, len(ids[2]) - 1)], ids[3][np.minimum(t, len(ids[3]) - 1)])
    code_keys = np.sort((ids[1][q].astype(np.int64) << 34) | (src.astype(np.int64) << 32) | tid.astype(np.int64))
    checks["submission_rows"] = rows
    checks["saved_codes_equal_submitted_file_pairs"] = bool(len(file_keys) == len(code_keys) and (file_keys == code_keys).all())
    # 2) baseline = rule(threshold 0.69, one-to-one) over all candidates
    checks["saved_codes_equal_rule_recomputation"] = bool(
        np.array_equal(cand[M.select(cand, np.asarray(p_all), BASE_T, True)], pred))
    checks["baseline_min_prob"] = float(p.min())
    checks["baseline_one_per_target"] = bool(len(np.unique(tkey(pred))) == len(pred))
    log(f"  checks {checks}")
    assert checks["saved_codes_equal_submitted_file_pairs"] and checks["saved_codes_equal_rule_recomputation"]

    # ------------------------------------------------------------------ France pair features / S1 attributes
    fi = np.flatnonzero(fr)
    idf_n, idf_a = (MF.IdfTable.load(TS / f"idf_test_{c}.npz") for c in ("name", "addr"))
    feat = {k: np.empty(len(fi), dtype=np.float32) for k in
            ("house_conflict", "name_cos", "addr_cos", "legal_only", "t_addr_missing", "eq_name_n2")}
    for a in range(0, len(fi), 1_000_000):
        X = MF.pair_features(S1, np.arange(n1), pools, pred[fi[a:a + 1_000_000]], idf_n, idf_a)
        sl = slice(a, a + X.shape[0])
        feat["house_conflict"][sl] = (X[:, C["house_both"]] == 1) & (X[:, C["eq_house"]] == 0)
        feat["name_cos"][sl] = X[:, C["name_cos"]]
        feat["addr_cos"][sl] = X[:, C["addr_cos"]]
        feat["legal_only"][sl] = (X[:, C["eq_name_n2"]] == 1) & (X[:, C["eq_name_n1"]] == 0)
        feat["t_addr_missing"][sl] = X[:, C["t_addr_missing"]]
        feat["eq_name_n2"][sl] = X[:, C["eq_name_n2"]]
    same_name_diff_addr = (feat["name_cos"] >= 0.8) & (feat["addr_cos"] < 0.3)
    fr_s1 = np.flatnonzero(c1 == CTRY["France"])
    name_n2 = np.asarray(S1.col("name_n2"))
    u, inv, cnt = np.unique(name_n2[fr_s1], return_inverse=True, return_counts=True)
    name_count = np.zeros(n1, dtype=np.int64)
    name_count[fr_s1] = cnt[inv]
    gen10 = name_count[q[fi]] >= 10
    gen100 = name_count[q[fi]] >= 100
    fp_, ftr, ftn, fq = p[fi], tr[fi], tn[fi], q[fi]
    base_npred = np.bincount(fq, minlength=n1)

    # rivals: would a full re-run of the one-per-target rule hand a removed France target to another S1?
    sel_all = np.flatnonzero(np.asarray(p_all) >= BASE_T)
    tk_all = tkey(cand[sel_all])
    q_all = cand[sel_all] >> B.Q_SHIFT
    p_sel = np.asarray(p_all[sel_all])

    # ------------------------------------------------------------------ simulation
    res = {"checks": checks, "baseline": {}, "thresholds": {}}
    n_fr_s1 = len(fr_s1)

    def fr_stats(keep_mask):
        npred = np.bincount(fq[keep_mask], minlength=n1)[fr_s1]
        g = name_count[fr_s1]
        return {"france_matches": int(keep_mask.sum()), "france_s1": n_fr_s1,
                "france_matches_per_s1": float(keep_mask.sum() / n_fr_s1),
                "france_s1_zero_predictions": int((npred == 0).sum()),
                "france_s1_zero_prediction_rate": float((npred == 0).mean()),
                "zero_pred_rate_generic10": float((npred[g >= 10] == 0).mean()),
                "zero_pred_rate_generic100": float((npred[g >= 100] == 0).mean()),
                "zero_pred_rate_unique": float((npred[g == 1] == 0).mean()),
                "pred_per_s1_generic10": float(npred[g >= 10].mean()),
                "pred_per_s1_unique": float(npred[g == 1].mean()),
                "share_house_conflict": float(feat["house_conflict"][keep_mask].mean()),
                "share_t_rank0_uncontested": float(((ftr == 0) & (ftn == 1))[keep_mask].mean())}

    allkeep = np.ones(len(fi), dtype=bool)
    res["baseline"] = {"total_predictions": int(len(pred)), **fr_stats(allkeep),
                       "us_india_predictions": int((~fr).sum())}
    removed_rows = []
    for th in THRESHOLDS:
        keep = fp_ >= th
        rm = ~keep
        sim_pred = np.concatenate([pred[~fr], pred[fi[keep]]])
        rm_tk = tkey(pred[fi[rm]])
        m = np.isin(tk_all, rm_tk)
        riv = m & ~np.isin(q_all, fq[rm])
        riv_ctry = c1[q_all[riv]]
        rm_s1 = np.unique(fq[rm])
        became_empty = rm_s1[np.bincount(fq[keep], minlength=n1)[rm_s1] == 0]
        loss = np.bincount(fq[rm], minlength=n1)[rm_s1]
        r = {"total_predictions": int(len(sim_pred)), **fr_stats(keep),
             "removed": int(rm.sum()), "removed_pct_of_france": float(rm.mean()),
             "removed_prob_distribution": prob_dist(fp_[rm], th),
             "removed_t_rank_distribution": rank_dist(ftr[rm]),
             "removed_t_rank0_uncontested": int(((ftr == 0) & (ftn == 1) & rm).sum()),
             "removed_t_ncand_mean": float(ftn[rm].mean()) if rm.any() else None,
             "removed_on_generic10_s1": int((gen10 & rm).sum()), "removed_on_generic100_s1": int((gen100 & rm).sum()),
             "removal_rate_generic10": float(rm[gen10].mean()), "removal_rate_unique": float(rm[name_count[fq] == 1].mean()),
             "removed_house_conflict": int((feat["house_conflict"].astype(bool) & rm).sum()),
             "removal_rate_house_conflict": float(rm[feat["house_conflict"] == 1].mean()),
             "removal_rate_no_house_conflict": float(rm[feat["house_conflict"] == 0].mean()),
             "removed_same_name_diff_address": int((same_name_diff_addr & rm).sum()),
             "removed_legal_form_only_name_diff": int((feat["legal_only"].astype(bool) & rm).sum()),
             "removed_target_address_empty": int((feat["t_addr_missing"].astype(bool) & rm).sum()),
             "france_s1_losing_predictions": int(len(rm_s1)),
             "france_s1_becoming_empty": int(len(became_empty)),
             "loss_per_affected_s1": {str(k): int((loss == k).sum()) for k in range(1, 6)} | {"6+": int((loss >= 6).sum())},
             "one_per_target_satisfied": bool(len(np.unique(tkey(sim_pred))) == len(sim_pred)),
             "us_india_predictions_identical": bool(np.array_equal(np.sort(sim_pred[:int((~fr).sum())]), np.sort(pred[~fr]))),
             "removed_targets_with_rival_claim_ge_0.69": int(np.isin(rm_tk, tk_all[riv]).sum()),
             "rival_claims_by_country": {c: int((riv_ctry == h).sum()) for c, h in CTRY.items()},
             "example_s1_becoming_empty": [f"S1-{int(ids[1][x])}" for x in became_empty[:15]]}
        res["thresholds"][str(th)] = r
        log(f"  t={th}: France {r['france_matches']:,} (removed {r['removed']:,}, {100 * r['removed_pct_of_france']:.2f}%), "
            f"zero-pred S1 {r['france_s1_zero_predictions']:,}; 1:1 ok {r['one_per_target_satisfied']}")
    # all pairs removed at the largest threshold (superset of the others), with the smallest threshold that removes them
    rm = fp_ < max(THRESHOLDS)
    with open(OUT / "removed_france_pairs_upto_0.88.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tremoved_entity_id\tprob\tremoved_from_threshold\tt_rank\tt_ncand\t"
                "house_conflict\tsame_name_diff_address\tlegal_form_only\ttarget_address_empty\ts1_name_count\n")
        for j in np.flatnonzero(rm):
            k = fi[j]
            tid_ = f"S{int(src[k])}-{int(ids[int(src[k])][t[k]])}"
            first = min(th for th in THRESHOLDS if fp_[j] < th)
            f.write(f"S1-{int(ids[1][q[k]])}\t{tid_}\t{fp_[j]:.4f}\t{first}\t{int(ftr[j])}\t{int(ftn[j])}\t"
                    f"{int(feat['house_conflict'][j])}\t{int(same_name_diff_addr[j])}\t{int(feat['legal_only'][j])}\t"
                    f"{int(feat['t_addr_missing'][j])}\t{int(name_count[q[k]])}\n")

    # a few raw examples (removed at 0.80), names/addresses from the test files
    ex_j = np.flatnonzero(fp_ < 0.80)
    ex_j = ex_j[np.argsort(fp_[ex_j], kind="stable")][:: max(1, len(ex_j) // 12)][:12]
    need = {1: {int(q[fi[j]]) for j in ex_j}, 2: set(), 3: set()}
    for j in ex_j:
        need[int(src[fi[j]])].add(int(t[fi[j]]))
    raw = {1: {}, 2: {}, 3: {}}
    for k in (1, 2, 3):
        if not need[k]:
            continue
        last = max(need[k])
        with open(ROOT / "dataset" / "test" / f"test_source{k}.tsv", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i in need[k]:
                    raw[k][i] = line.rstrip("\n").split("\t")[1:3]
                if i >= last:
                    break
    res["examples_removed_at_0.80"] = [
        {"prob": float(fp_[j]), "t_rank": int(ftr[j]), "t_ncand": int(ftn[j]), "s1": raw[1][int(q[fi[j]])],
         "target": raw[int(src[fi[j]])][int(t[fi[j]])], "house_conflict": int(feat["house_conflict"][j]),
         "s1_name_count": int(name_count[q[fi[j]]])} for j in ex_j]

    # ------------------------------------------------------------------ labelled comparison (validation; no France)
    from blocking_experiment import Context
    ctx = Context(ROOT / "experiments" / "blocking" / "features", ROOT / "experiments" / "validation_split",
                  ROOT / "experiments" / "_eda_cache")
    qv = np.flatnonzero(ctx.fold[1] == 1)
    cv = np.asarray(ctx.s[1].col("country"))[qv]
    true = ctx.truth(qv)
    nq = len(qv)
    lab = {}
    envs = {"V": MD / "cand_val.npy", "T": TD / "cand_T.npy"}
    cases = [("current model, US (in-distribution)", "A", "US", BASE_T),
             ("current model, India (in-distribution)", "A", "India", BASE_T),
             ("LOCO over-merge analogue: India-only model on US", "IN", "US", 0.68),
             ("LOCO under-merge analogue: US-only model on India", "US", "India", 0.74)]
    for env, cp in envs.items():
        codes = np.load(cp)
        qq = codes >> B.Q_SHIFT
        lab[env] = {}
        for name, mdl, ec, base in cases:
            pm = (cv == CTRY[ec])[qq]
            prob = np.load(TD / f"p_{env}_A.npy") if mdl == "A" else np.load(LD / f"p_{env}_{mdl}.npy")
            cs, ps = codes[pm], prob[pm]
            keep = M.select(cs, ps, base, True)
            bc, bp = cs[keep], ps[keep]
            qmask = cv == CTRY[ec]
            tc = true[qmask[true >> B.Q_SHIFT]]
            rows = {}
            for th in [base] + THRESHOLDS:
                k2 = bp >= th
                _, parts = M.entity_f05(bc[k2], tc, nq, return_parts=True)
                F = parts["F"][qmask]
                single = parts["ntrue"][qmask] == 0
                rmv = bc[~k2]
                rows[str(th)] = {"macro_f05": float(F.mean()), "singleton_f05": float(F[single].mean()),
                                 "predictions": int(k2.sum()), "removed": int((~k2).sum()),
                                 "removed_that_were_true_matches": int(np.isin(rmv, tc, assume_unique=True).sum()),
                                 "zero_prediction_rate": float((parts["npred"][qmask] == 0).mean())}
            lab[env][name] = rows
            log(f"  labelled {env} | {name}: " + ", ".join(f"{k}:{v['macro_f05']:.4f}" for k, v in rows.items()))
    res["labelled_comparison"] = lab

    hashes_after = {f: sha256(ROOT / f) for f in PROTECTED}
    res["integrity"] = {"protected_files_unchanged": all(hashes_after[f] == hashes_before[f] for f in PROTECTED),
                        "n_files": len(PROTECTED),
                        "output/matching_results.tsv": hashes_after["output/matching_results.tsv"],
                        "output/candidate_pairs.tsv": hashes_after["output/candidate_pairs.tsv"]}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"  integrity {res['integrity']}; done in {res['seconds']}s")


if __name__ == "__main__":
    main()
