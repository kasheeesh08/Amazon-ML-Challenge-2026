#!/usr/bin/env python3
"""Deep France error analysis (READ-ONLY diagnostic; no labels exist for France; no France F0.5 is produced).

Inputs (all existing): validation labels + saved candidates/features/probabilities of the labelled environments
V (normal), T (test-ratio), S (full-pool stress) from experiments/matching and experiments/diagnostics/trank; LOCO
cross-country probabilities (experiments/diagnostics/leave_one_country_out); the real test candidates, target context
and Step 3-model probabilities (experiments/test_scale) and the submitted final-model probabilities/predictions
(experiments/final); raw names from dataset/train and dataset/test (read-only, for text-level probes).

Outputs: experiments/diagnostics/france_error_analysis/{results.json, run.log, hashes_before.json}.
"""

import json
import re
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
from ber import normalize as N  # noqa: E402
from ber.features import FeatureStore  # noqa: E402
from ber.normalize import h64  # noqa: E402
from test_scale_check import sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
MD = ROOT / "experiments" / "matching"
TD = ROOT / "experiments" / "diagnostics" / "trank"
LD = ROOT / "experiments" / "diagnostics" / "leave_one_country_out"
SD = ROOT / "experiments" / "test_scale" / "stress"
TS = ROOT / "experiments" / "test_scale"
FD = ROOT / "experiments" / "final"
C = MF.COL
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
PBINS = [(0.69, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
PBIN_NAMES = ["[0.69,0.80)", "[0.80,0.90)", "[0.90,0.95)", ">=0.95"]
ACR = re.compile(r"[\s\.\-&'/,()]")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def is_acronym(name):
    n = ACR.sub("", name)
    return 2 <= len(n) <= 4 and n.isalpha() and n.isupper()


def acronym_flags(tsv, n_rows):
    """Per file row: target name is a 2-4 letter initialism (e.g. 'SC', 'L.C.')."""
    out = np.zeros(n_rows, dtype=bool)
    with open(tsv, encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            a = line.find("\t")
            b = line.find("\t", a + 1)
            out[i] = is_acronym(line[a + 1:b])
    return out


def pct(x, qs=(5, 50, 90, 99)):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {}
    return {"mean": float(x.mean()), **{f"p{q}": float(v) for q, v in zip(qs, np.percentile(x, qs))}}


def name_freq(names, mask):
    f = np.zeros(len(names), dtype=np.int64)
    _, inv, cnt = np.unique(names[mask], return_inverse=True, return_counts=True)
    f[mask] = cnt[inv]
    return f


def prob_bin(p):
    """0: below 0.80 (includes [0.68,0.69) for the India-only LOCO model's 0.68 threshold), 1: [0.80,0.90),
    2: [0.90,0.95), 3: >= 0.95."""
    return np.digitize(np.asarray(p), [0.80, 0.90, 0.95]).astype(np.int64)


def pred_categories(p, hc, legal, unc, addr_empty, acro, house_absent):
    return {"house-number conflict": hc, "house number absent on either side": house_absent,
            "legal-form-only name difference": legal, "t_rank0 & uncontested target": unc,
            "target address empty": addr_empty, "target name is an initialism": acro,
            **{f"prob {n}": prob_bin(p) == i for i, n in enumerate(PBIN_NAMES)}}


def cell_ids(p, hc, legal, unc, addr_empty, acro):
    return prob_bin(p) * 32 + hc * 16 + legal * 8 + unc * 4 + addr_empty * 2 + acro


def mix_precision(src_cells, src_y, dst_cells, min_n=30):
    """Precision implied by the destination's cell mix using the source's per-cell precision (fallback: prob bin)."""
    prec = {}
    for c in np.unique(src_cells):
        m = src_cells == c
        prec[int(c)] = (float(src_y[m].mean()), int(m.sum()))
    binp = {b: float(src_y[(src_cells // 32) == b].mean()) for b in range(4) if ((src_cells // 32) == b).any()}
    vals = np.empty(len(dst_cells))
    for c in np.unique(dst_cells):
        m = dst_cells == c
        pr, n = prec.get(int(c), (None, 0))
        vals[m] = pr if n >= min_n else binp[int(c) // 32]
    return float(vals.mean())


def main():
    t0 = time.time()
    hb = {f: sha256(ROOT / f) for f in PROTECTED}
    hb_final = hash_tree(FD)
    (OUT / "hashes_before.json").write_text(json.dumps({"protected": hb, "experiments_final": hb_final}, indent=1),
                                            encoding="utf-8")
    res = {"profiles": {}, "labelled_metrics": {}, "prediction_categories": {}, "reweighted_us": {},
           "mix_estimator_validation": {}, "france_mix_implied_precision": {}, "france_actual": {}, "pool": {}}

    # ------------------------------------------------------------------ stores and raw-text flags
    from blocking_experiment import Context
    ctx = Context(ROOT / "experiments" / "blocking" / "features", ROOT / "experiments" / "validation_split",
                  ROOT / "experiments" / "_eda_cache")
    S1t = FeatureStore(TS / "features_test", "test_s1")
    Pt = {k: FeatureStore(TS / "features_test", f"test_s{k}") for k in (2, 3)}
    acr_train = {k: acronym_flags(ROOT / "dataset" / "train" / f"train_source{k}.tsv", ctx.s[k].n) for k in (2, 3)}
    acr_test = {k: acronym_flags(ROOT / "dataset" / "test" / f"test_source{k}.tsv", Pt[k].n) for k in (2, 3)}
    log("  initialism flags computed")

    def pool_props(stores, rows_by_src, acr, country):
        props = {}
        tot = 0
        agg = {"address empty": 0, "name non-Latin": 0, "postal present": 0, "house number present": 0,
               "name is an initialism": 0}
        for k in (2, 3):
            r = rows_by_src[k]
            cm = np.asarray(stores[k].col("country"))[r] == CTRY[country]
            rr = r[cm]
            fl = np.asarray(stores[k].col("flags"))[rr].astype(np.int64)
            tot += len(rr)
            agg["address empty"] += int(((fl & N.FLAG_ADDR_MISSING) > 0).sum())
            agg["name non-Latin"] += int(((fl & N.FLAG_NAME_NONLATIN) > 0).sum())
            agg["postal present"] += int(((fl & N.FLAG_POSTAL_PRESENT) > 0).sum())
            agg["house number present"] += int((np.asarray(stores[k].col("house"))[rr] != 0).sum())
            agg["name is an initialism"] += int(acr[k][rr].sum())
        props = {k: v / tot for k, v in agg.items()}
        props["records"] = tot
        return props

    qv = np.flatnonzero(ctx.fold[1] == 1)
    nq = len(qv)
    cv = np.asarray(ctx.s[1].col("country"))[qv]
    true = ctx.truth(qv)
    ntrue = np.bincount(true >> B.Q_SHIFT, minlength=nq)
    names_v = np.asarray(ctx.s[1].col("name_n2"))[qv]
    house_v = np.asarray(ctx.s[1].col("house"))[qv]
    freq_v = np.zeros(nq, dtype=np.int64)
    for c in ("US", "India"):
        freq_v += name_freq(names_v, cv == CTRY[c])
    pools = {"V": {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)},
             "T": {k: np.load(TD / f"poolT_s{k}.npy") for k in (2, 3)},
             "S": {k: np.arange(ctx.s[k].n) for k in (2, 3)}}
    for env in ("V", "T", "S"):
        for c in ("US", "India"):
            res["pool"][f"{env}-{c}"] = pool_props(ctx.s, pools[env], acr_train, c)
    for c in CTRY:
        res["pool"][f"TEST-{c}"] = pool_props(Pt, {k: np.arange(Pt[k].n) for k in (2, 3)}, acr_test, c)
    log("  pool properties done")

    # ------------------------------------------------------------------ labelled groups
    envs = {"V": (MD / "cand_val.npy", MD / "X_val.npy"), "T": (TD / "cand_T.npy", TD / "X_T.npy"),
            "S": (SD / "cand_stress.npy", TD / "X_S.npy")}
    groups = [(e, c, "A", 0.69) for e in ("V", "T", "S") for c in ("US", "India")] + \
             [("T", "US", "IN", 0.68), ("T", "India", "US", 0.74), ("T", "India", "IN", 0.68), ("T", "US", "US", 0.74)]
    cells_store = {}
    for env, (cp, xp) in envs.items():
        codes = np.load(cp)
        q = codes >> B.Q_SHIFT
        _, s_all, t_all = B.decode(codes)
        y = np.isin(codes, true, assume_unique=True)
        cols = np.load(TD / f"cols_{env}.npz")
        X = np.load(xp, mmap_mode="r")
        cnt = np.bincount(q, minlength=nq)
        for (e, c, mdl, th) in [g for g in groups if g[0] == env]:
            tag = f"{env}-{c}-{'Step3' if mdl == 'A' else mdl + '-model'}"
            pm = (cv == CTRY[c])[q]
            ii = np.flatnonzero(pm)
            prob = np.load(TD / f"p_{env}_A.npy") if mdl == "A" else np.load(LD / f"p_{env}_{mdl}.npy")
            keep = M.select(codes[ii], prob[ii], th, True)
            pi = ii[keep]
            qmask = cv == CTRY[c]
            tc = true[qmask[true >> B.Q_SHIFT]]
            _, parts = M.entity_f05(codes[pi], tc, nq, return_parts=True)
            F, P, R, npred = parts["F"][qmask], parts["P"][qmask], parts["R"][qmask], parts["npred"][qmask]
            nt = ntrue[qmask]
            fq = freq_v[qmask]
            yy = y[pi]
            pp = prob[pi]
            xr = np.asarray(X[pi][:, [C["eq_name_n1"], C["eq_name_n2"], C["t_gap"]]])
            hc = ((cols["house_both"][pi] == 1) & (cols["eq_house"][pi] == 0)).astype(np.int64)
            house_absent = (cols["house_both"][pi] == 0)
            legal = ((xr[:, 1] == 1) & (xr[:, 0] == 0)).astype(np.int64)
            unc = ((cols["t_rank"][pi] == 0) & (cols["t_ncand"][pi] == 1)).astype(np.int64)
            ae = (cols["t_addr_missing"][pi] == 1).astype(np.int64)
            acro = np.zeros(len(pi), dtype=np.int64)
            for k in (2, 3):
                mk = s_all[pi] == k
                acro[mk] = acr_train[k][t_all[pi][mk]]
            cats = pred_categories(pp, hc.astype(bool), legal.astype(bool), unc.astype(bool), ae.astype(bool),
                                   acro.astype(bool), house_absent)
            tp = int(yy.sum())
            res["labelled_metrics"][tag] = {
                "macro_f05": float(F.mean()), "pair_precision": tp / len(pi), "pair_recall": tp / len(tc),
                "entity_precision": float(P[npred > 0].mean()), "entity_recall": float(R[nt > 0].mean()),
                "fp": int(len(pi) - tp), "fn": int(len(tc) - tp), "fn_not_in_candidates": int(len(tc) - y[ii].sum()),
                "candidate_recall": float(y[ii].sum() / len(tc)),
                "singleton_f05": float(F[nt == 0].mean()), "singleton_false_merge_rate": float((npred[nt == 0] > 0).mean()),
                "zero_prediction_rate": float((npred == 0).mean()), "pred_per_s1": float(npred.mean()),
                "f05_generic10_s1": float(F[fq >= 10].mean()), "f05_generic100_s1": float(F[fq >= 100].mean()) if (fq >= 100).any() else None,
                "f05_unique_s1": float(F[fq == 1].mean()),
                "fp_prob_ge_0.95": int(((pp >= 0.95) & ~yy).sum()),
                "share_of_fp_with_prob_ge_0.95": float(((pp >= 0.95) & ~yy).sum() / max(1, (~yy).sum())),
                "precision_by_category": {k: (float(yy[m].mean()) if m.any() else None) for k, m in cats.items()},
                "recall_empty_address_targets": None}
            res["prediction_categories"][tag] = {k: float(m.mean()) for k, m in cats.items()}
            if mdl == "A" and env == "T":
                # recall on empty-address true pairs (target flag), candidate stage + end-to-end
                tq_, ts_, tt_ = B.decode(tc)
                fl = np.empty(len(tc), dtype=np.int64)
                for k in (2, 3):
                    mk = ts_ == k
                    fl[mk] = np.asarray(ctx.s[k].col("flags"))[tt_[mk]]
                emp = (fl & N.FLAG_ADDR_MISSING) > 0
                res["labelled_metrics"][tag]["recall_empty_address_targets"] = {
                    "candidate": float(np.isin(tc[emp], codes[ii], assume_unique=True).mean()),
                    "end_to_end": float(np.isin(tc[emp], codes[pi], assume_unique=True).mean())}
            # profile (observable, label-free)
            cc = cnt[qmask]
            res["profiles"][tag] = {
                "cand_per_s1": pct(cc), "share_s1_ge_300_candidates": float((cc >= 300).mean()),
                "share_s1_name_shared_10+": float((fq >= 10).mean()), "share_s1_name_shared_100+": float((fq >= 100).mean()),
                "share_s1_house_present": float((house_v[qmask] != 0).mean()),
                "cand_share_t_rank0": float((cols["t_rank"][ii] == 0).mean()),
                "cand_t_ncand": pct(cols["t_ncand"][ii]), "pred_t_gap_p50": float(np.median(xr[:, 2])),
                "pred_per_s1": float(npred.mean()), "zero_prediction_rate": float((npred == 0).mean()),
                "true_singleton_rate": float((nt == 0).mean())}
            cells_store[tag] = (cell_ids(pp, hc, legal, unc, ae, acro), yy)
            # France-profile reweighting inputs (Step3, US)
            if mdl == "A" and c == "US":
                res["reweighted_us"][env] = {"F": F, "fq": fq, "house": house_v[qmask] != 0}
            log(f"  {tag}: F0.5 {F.mean():.5f} P {tp / len(pi):.4f} R {tp / len(tc):.4f}")
        del codes, q, y, X

    # ------------------------------------------------------------------ test (unlabelled): Step 3 model + final model
    n1 = S1t.n
    c1 = np.asarray(S1t.col("country"))
    names_t = np.asarray(S1t.col("name_n2"))
    house_t = np.asarray(S1t.col("house"))
    freq_t = np.zeros(n1, dtype=np.int64)
    for c in CTRY:
        freq_t += name_freq(names_t, c1 == CTRY[c])
    cand = np.load(TS / "cand_test.npy")
    qt = cand >> B.Q_SHIFT
    cnt_t = np.bincount(qt, minlength=n1)
    tr_all = np.load(TS / "t_rank_test.npy", mmap_mode="r")
    tn_all = np.load(TS / "t_ncand_test.npy", mmap_mode="r")
    tg_all = np.load(TS / "t_gap_test.npy", mmap_mode="r")
    p3_all = np.load(TS / "prob_test.npy")
    pf_all = np.load(FD / "test_prob.npy", mmap_mode="r")
    idx = {"Step3": np.flatnonzero(M.select(cand, p3_all, 0.69, True))}
    predf = np.load(FD / "test_pred_codes.npy")
    idx["final"] = np.searchsorted(cand, predf)
    assert (cand[idx["final"]] == predf).all()
    idf_n, idf_a = (MF.IdfTable.load(TS / f"idf_test_{c}.npz") for c in ("name", "addr"))
    for mdl, ix in idx.items():
        pr = (p3_all if mdl == "Step3" else pf_all)
        pp_all = np.asarray(pr[ix])
        qx = qt[ix]
        _, sx, tx = B.decode(cand[ix])
        hc = np.empty(len(ix), dtype=np.int64)
        legal = np.empty(len(ix), dtype=np.int64)
        ae = np.empty(len(ix), dtype=np.int64)
        habs = np.empty(len(ix), dtype=bool)
        for a in range(0, len(ix), 1_000_000):
            Xb = MF.pair_features(S1t, np.arange(n1), Pt, cand[ix[a:a + 1_000_000]], idf_n, idf_a)
            sl = slice(a, a + Xb.shape[0])
            hc[sl] = (Xb[:, C["house_both"]] == 1) & (Xb[:, C["eq_house"]] == 0)
            habs[sl] = Xb[:, C["house_both"]] == 0
            legal[sl] = (Xb[:, C["eq_name_n2"]] == 1) & (Xb[:, C["eq_name_n1"]] == 0)
            ae[sl] = Xb[:, C["t_addr_missing"]]
        tr = np.asarray(tr_all[ix])
        tn = np.asarray(tn_all[ix])
        tg = np.asarray(tg_all[ix])
        unc = ((tr == 0) & (tn == 1)).astype(np.int64)
        acro = np.zeros(len(ix), dtype=np.int64)
        for k in (2, 3):
            mk = sx == k
            acro[mk] = acr_test[k][tx[mk]]
        npred_all = np.bincount(qx, minlength=n1)
        for c in CTRY:
            m = c1[qx] == CTRY[c]
            sm = c1 == CTRY[c]
            tag = f"TEST-{c}-{mdl}"
            cats = pred_categories(pp_all[m], hc[m].astype(bool), legal[m].astype(bool), unc[m].astype(bool),
                                   ae[m].astype(bool), acro[m].astype(bool), habs[m])
            res["prediction_categories"][tag] = {k: float(v.mean()) for k, v in cats.items()}
            npd = npred_all[sm]
            fq = freq_t[sm]
            cc = cnt_t[sm]
            res["profiles"][tag] = {
                "cand_per_s1": pct(cc), "share_s1_ge_300_candidates": float((cc >= 300).mean()),
                "share_s1_name_shared_10+": float((fq >= 10).mean()), "share_s1_name_shared_100+": float((fq >= 100).mean()),
                "share_s1_house_present": float((house_t[sm] != 0).mean()),
                "cand_share_t_rank0": None, "cand_t_ncand": None, "pred_t_gap_p50": float(np.median(tg[m])),
                "pred_per_s1": float(npd.mean()), "zero_prediction_rate": float((npd == 0).mean()),
                "true_singleton_rate": None}
            if c == "France":
                res["france_actual"][mdl] = {
                    "s1": int(sm.sum()), "matches": int(m.sum()), "matches_per_s1": float(npd.mean()),
                    "no_match_s1": int((npd == 0).sum()), "no_match_rate": float((npd == 0).mean()),
                    "candidates_per_s1": pct(cc, (1, 5, 50, 90, 95, 99)) | {"max": int(cc.max())},
                    "prob_bins": {n: float((prob_bin(pp_all[m]) == i).mean()) for i, n in enumerate(PBIN_NAMES)},
                    "generic_name_rate_s1_10+": float((fq >= 10).mean()), "generic_name_rate_s1_100+": float((fq >= 100).mean()),
                    "s1_house_present_rate": float((house_t[sm] != 0).mean()),
                    "pred_house_conflict_rate": float(hc[m].mean()), "pred_house_absent_rate": float(habs[m].mean()),
                    "pred_target_address_empty_rate": float(ae[m].mean()),
                    "pred_initialism_target_rate": float(acro[m].mean()),
                    "pred_t_rank": pct(tr[m]), "pred_t_gap": pct(tg[m]), "pred_t_ncand": pct(tn[m]),
                    "pred_t_rank0_uncontested": float(unc[m].mean()),
                    "no_match_rate_generic10_vs_unique": [float((npd[fq >= 10] == 0).mean()), float((npd[fq == 1] == 0).mean())]}
            if mdl == "Step3":
                cells_store[tag] = (cell_ids(pp_all[m], hc[m], legal[m], unc[m], ae[m], acro[m]), None)
        log(f"  test {mdl} categories done")
    # candidate-level t features on test by country (every 5th pair)
    sub = slice(None, None, 5)
    ctry_pairs = c1[qt[sub]]
    for c in CTRY:
        m = ctry_pairs == CTRY[c]
        res["profiles"][f"TEST-{c}-Step3"]["cand_share_t_rank0"] = float((np.asarray(tr_all[sub])[m] == 0).mean())
        res["profiles"][f"TEST-{c}-Step3"]["cand_t_ncand"] = pct(np.asarray(tn_all[sub])[m])
    del cand, qt, p3_all

    # ------------------------------------------------------------------ mix-implied precision: validation of the estimator
    val = {}
    for name, src, dst in (("India in-country cells -> India-only model on US (over-merge, actual cross-country)", "T-India-IN-model", "T-US-IN-model"),
                           ("US in-country cells -> US-only model on India (under-merge, actual cross-country)", "T-US-US-model", "T-India-US-model"),
                           ("Step3 US cells -> Step3 India (same model, other country)", "T-US-Step3", "T-India-Step3"),
                           ("Step3 T-US cells -> Step3 S-US (orphan-heavy)", "T-US-Step3", "S-US-Step3")):
        est = mix_precision(*cells_store[src], cells_store[dst][0])
        act = float(cells_store[dst][1].mean())
        val[name] = {"estimated_precision": est, "actual_precision": act, "error": est - act}
    res["mix_estimator_validation"] = val
    for src in ("T-US-Step3", "T-India-Step3"):
        res["france_mix_implied_precision"][f"cells from {src}"] = mix_precision(*cells_store[src], cells_store["TEST-France-Step3"][0])
        res["france_mix_implied_precision"][f"same cells applied to TEST-US (reference)"] = mix_precision(*cells_store[src], cells_store["TEST-US-Step3"][0])
    res["france_mix_implied_precision"]["cells from T-US-Step3 -> TEST-India"] = mix_precision(*cells_store["T-US-Step3"], cells_store["TEST-India-Step3"][0])

    # ------------------------------------------------------------------ France-profile reweighting of US validation S1
    fr = c1 == CTRY["France"]
    fq_fr = freq_t[fr]
    h_fr = house_t[fr] != 0

    def bucket(fq, h):
        fb = np.digitize(fq, [2, 10, 100])  # 1 | 2-9 | 10-99 | 100+
        return fb * 2 + h.astype(np.int64)
    bf = np.bincount(bucket(fq_fr, h_fr), minlength=8) / fr.sum()
    rw = {"France_cell_shares": bf.tolist()}
    for env, d in res["reweighted_us"].items():
        bu = bucket(d["fq"], d["house"])
        us_sh = np.bincount(bu, minlength=8) / len(bu)
        w = np.where(us_sh[bu] > 0, bf[bu] / np.maximum(us_sh[bu], 1e-12), 0.0)
        rw[env] = {"US_cell_shares": us_sh.tolist(), "unweighted_f05": float(d["F"].mean()),
                   "france_profile_weighted_f05": float((w * d["F"]).sum() / w.sum()),
                   "uncovered_france_share": float(bf[us_sh == 0].sum())}
    res["reweighted_us"] = rw

    ha = {f: sha256(ROOT / f) for f in PROTECTED}
    ha_final = hash_tree(FD)
    res["integrity"] = {"protected_unchanged": ha == hb, "n_protected": len(hb),
                        "experiments_final_unchanged": ha_final == hb_final, "n_final_files": len(hb_final),
                        "output/matching_results.tsv": ha["output/matching_results.tsv"],
                        "output/candidate_pairs.tsv": ha["output/candidate_pairs.tsv"]}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  integrity {res['integrity']}; total {res['seconds']}s")


if __name__ == "__main__":
    main()
