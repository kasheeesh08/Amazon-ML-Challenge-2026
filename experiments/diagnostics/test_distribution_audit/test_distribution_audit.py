#!/usr/bin/env python3
"""Train-vs-test distribution audit (READ-ONLY; no retraining, no inference, no new submission).

Uses only saved artifacts:
  training / validation stores + folds + ground truth (experiments/blocking/features, validation_split, _eda_cache)
  labelled environments V / T (candidates: experiments/matching/cand_val.npy, diagnostics/trank/cand_T.npy;
    Step 3-model probabilities diagnostics/trank/p_V_A.npy, p_T_A.npy; S candidates for blocking losses)
  real test: experiments/test_scale/{cand_test.npy, prob_test.npy (Step 3 model), features_test/}
             experiments/final/{test_prob.npy, test_pred_codes.npy} (submitted final model)
  earlier diagnostic results (read, not rerun): diagnostics/blocking_gap/results.json,
    diagnostics/france_error_analysis/results.json
Outputs: experiments/diagnostics/test_distribution_audit/{results.json, test_matched_subset_s1_ids.txt, run.log}
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
from ber import matcher as M  # noqa: E402
from ber.features import FeatureStore  # noqa: E402
from ber.normalize import h64  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402
from test_scale_check import sha256  # noqa: E402

OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
TD, MD, TS, FD, SD = (EXP / "diagnostics" / "trank", EXP / "matching", EXP / "test_scale", EXP / "final",
                      EXP / "test_scale" / "stress")
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
NP_BUCKETS = [0, 1, 2, 3, 4, 5, 6]          # 6 = 6+
THS = [0.69, 0.80, 0.90, 0.95]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def npb(n):
    return np.minimum(n, 6).astype(np.int64)


def pred_stats(codes_pred, probs_pred, nq, s1_mask, n_true=None):
    q = codes_pred >> B.Q_SHIFT
    _, s, _ = B.decode(codes_pred)
    npred = np.bincount(q, minlength=nq)[s1_mask]
    m = s1_mask[q]
    r = {"s1": int(s1_mask.sum()), "predictions": int(m.sum()), "pred_per_s1": float(npred.mean()),
         "empty_prediction_rate": float((npred == 0).mean()),
         "npred_distribution": {str(b): float((npb(npred) == b).mean()) for b in NP_BUCKETS},
         "pred_share_S2": float((s[m] == 2).mean()),
         "pred_prob_quantiles": {f"p{k}": float(v) for k, v in zip((5, 25, 50), np.percentile(probs_pred[m], (5, 25, 50)))},
         "pred_prob_bins": {"[0.69,0.80)": float(((probs_pred[m] >= 0.69) & (probs_pred[m] < 0.8)).mean()),
                            "[0.80,0.90)": float(((probs_pred[m] >= 0.8) & (probs_pred[m] < 0.9)).mean()),
                            "[0.90,0.95)": float(((probs_pred[m] >= 0.9) & (probs_pred[m] < 0.95)).mean()),
                            ">=0.95": float((probs_pred[m] >= 0.95).mean())}}
    if n_true is not None:
        nt = n_true[s1_mask]
        r["true_matches_per_s1"] = float(nt.mean())
        r["true_singleton_rate"] = float((nt == 0).mean())
        r["true_distribution"] = {str(b): float((npb(nt) == b).mean()) for b in NP_BUCKETS}
    return r, npred


def cand_stats(codes, probs, nq, s1_mask):
    q = codes >> B.Q_SHIFT
    cnt = np.bincount(q, minlength=nq)[s1_mask]
    m = s1_mask[q]
    pm = probs[m]
    starts = np.searchsorted(q, np.arange(nq))
    has = np.diff(np.append(starts, len(q))) > 0
    mx = np.full(nq, -1.0, dtype=np.float32)
    mx[has] = np.maximum.reduceat(probs, starts[has])
    mxs = mx[s1_mask]
    return {"cand_per_s1": {"mean": float(cnt.mean()), **{f"p{k}": float(v) for k, v in
                                                           zip((50, 90, 95, 99), np.percentile(cnt, (50, 90, 95, 99)))}},
            "cand_share_ge": {str(t): float((pm >= t).mean()) for t in THS},
            "s1_max_prob_quantiles": {f"p{k}": float(v) for k, v in zip((1, 5, 10, 50), np.percentile(mxs, (1, 5, 10, 50)))},
            "s1_max_prob_lt_0.10": float((mxs < 0.10).mean()),
            "s1_max_prob_in_[0.10,0.69)": float(((mxs >= 0.10) & (mxs < 0.69)).mean())}


def main():
    t0 = time.time()
    watch = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "experiments/final/model_final.npz",
             "camelcase_submission.zip", "experiments/blocking/blocking_config.json"]
    hb = {f: sha256(ROOT / f) for f in watch}
    res = {"hashes_before": hb}

    from blocking_experiment import Context
    ctx = Context(EXP / "blocking" / "features", EXP / "validation_split", EXP / "_eda_cache")
    S1t = FeatureStore(TS / "features_test", "test_s1")
    Pt = {k: FeatureStore(TS / "features_test", f"test_s{k}") for k in (2, 3)}

    # ------------------------------------------------------------ 1-2. S1 and pool composition
    c1 = np.asarray(ctx.s[1].col("country"))
    fold1 = ctx.fold[1]
    c1t = np.asarray(S1t.col("country"))
    comp = {"S1": {}, "pool": {}}
    for c, h in CTRY.items():
        comp["S1"][c] = {"train_all": int((c1 == h).sum()), "fit": int(((c1 == h) & (fold1 == 0)).sum()),
                         "validation": int(((c1 == h) & (fold1 == 1)).sum()), "test": int((c1t == h).sum())}
    for k in (2, 3):
        ck = np.asarray(ctx.s[k].col("country"))
        ckt = np.asarray(Pt[k].col("country"))
        is_t = np.zeros(ctx.s[k].n, dtype=bool)
        is_t[ctx.pair_trow[ctx.pair_src == k]] = True
        for c, h in CTRY.items():
            d = comp["pool"].setdefault(c, {})
            d[f"S{k}_train_all"] = int((ck == h).sum())
            d[f"S{k}_validation_pool"] = int(((ck == h) & (ctx.fold[k] == 1)).sum())
            d[f"S{k}_test"] = int((ckt == h).sum())
            d[f"S{k}_train_true_targets"] = int(((ck == h) & is_t).sum())
    for c in CTRY:
        p, s = comp["pool"][c], comp["S1"][c]
        tr_pool = p["S2_train_all"] + p["S3_train_all"]
        te_pool = p["S2_test"] + p["S3_test"]
        comp["pool"][c]["pool_per_s1_train"] = tr_pool / s["train_all"] if s["train_all"] else None
        comp["pool"][c]["pool_per_s1_validation"] = ((p["S2_validation_pool"] + p["S3_validation_pool"]) / s["validation"]) if s["validation"] else None
        comp["pool"][c]["pool_per_s1_test"] = te_pool / s["test"]
        comp["pool"][c]["S2_share_train"] = p["S2_train_all"] / tr_pool if tr_pool else None
        comp["pool"][c]["S2_share_test"] = p["S2_test"] / te_pool
        comp["pool"][c]["train_true_target_share_of_pool"] = ((p["S2_train_true_targets"] + p["S3_train_true_targets"]) / tr_pool) if tr_pool else None
    res["composition"] = comp
    log("  composition done")

    # ------------------------------------------------------------ labelled environments V / T (Step 3 model)
    qv = np.flatnonzero(fold1 == 1)
    nq = len(qv)
    cv = c1[qv]
    true = ctx.truth(qv)
    ntrue = np.bincount(true >> B.Q_SHIFT, minlength=nq)
    lab = {}
    npred_env = {}
    for env, cp in (("V", MD / "cand_val.npy"), ("T", TD / "cand_T.npy")):
        codes = np.load(cp)
        p = np.load(TD / f"p_{env}_A.npy")
        keep = M.select(codes, p, 0.69, True)
        pc, pp = codes[keep], p[keep]
        _, parts = M.entity_f05(pc, true, nq, return_parts=True)
        lab[env] = {}
        npred_env[env] = np.bincount(pc >> B.Q_SHIFT, minlength=nq)
        pool_n = {"V": {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)},
                  "T": {k: np.load(TD / f"poolT_s{k}.npy") for k in (2, 3)}}[env]
        for c in ("US", "India"):
            m = cv == CTRY[c]
            st, _ = pred_stats(pc, pp, nq, m, ntrue)
            st.update(cand_stats(codes, p, nq, m))
            st["macro_f05"] = float(parts["F"][m].mean())
            npool = sum(int((np.asarray(ctx.s[k].col("country"))[pool_n[k]] == CTRY[c]).sum()) for k in (2, 3))
            st["pool_per_s1"] = npool / int(m.sum())
            st["share_of_pool_records_predicted"] = st["predictions"] / npool
            st["share_of_pool_records_that_are_true_targets"] = int(ntrue[m].sum()) / npool
            lab[env][c] = st
        log(f"  labelled {env} done")
    res["labelled"] = lab

    # ------------------------------------------------------------ real test (Step 3 model and final model)
    n1 = S1t.n
    cand = np.load(TS / "cand_test.npy")
    p3 = np.load(TS / "prob_test.npy")
    pf = np.load(FD / "test_prob.npy")
    keep3 = M.select(cand, p3, 0.69, True)
    predf = np.load(FD / "test_pred_codes.npy")
    idxf = np.searchsorted(cand, predf)
    assert (cand[idxf] == predf).all()
    test = {"Step3": {}, "final": {}}
    npred_test = {}
    pool_ct = {c: sum(int((np.asarray(Pt[k].col("country")) == h).sum()) for k in (2, 3)) for c, h in CTRY.items()}
    for tag, pcodes, pprob, allp in (("Step3", cand[keep3], p3[keep3], p3), ("final", predf, pf[idxf], pf)):
        npred_test[tag] = np.bincount(pcodes >> B.Q_SHIFT, minlength=n1)
        for c, h in CTRY.items():
            m = c1t == h
            st, _ = pred_stats(pcodes, pprob, n1, m)
            st.update(cand_stats(cand, allp, n1, m))
            st["pool_per_s1"] = pool_ct[c] / int(m.sum())
            st["share_of_pool_records_predicted"] = st["predictions"] / pool_ct[c]
            test[tag][c] = st
        log(f"  test {tag} done")
    res["test"] = test
    del cand, p3, pf, keep3

    # ------------------------------------------------------------ earlier diagnostics (read only)
    bg = json.loads((EXP / "diagnostics" / "blocking_gap" / "results.json").read_text(encoding="utf-8"))
    fe = json.loads((EXP / "diagnostics" / "france_error_analysis" / "results.json").read_text(encoding="utf-8"))
    res["from_earlier_diagnostics"] = {
        "test_generic_names_by_country": {c: v["generic_names"] for c, v in bg["test"]["by_country"].items()},
        "test_pool_flag_rates": {c: v["pool_flag_rates"] for c, v in bg["test"]["by_country"].items()},
        "train_pool_flag_rates": bg["test"]["train_pool_flag_rates"],
        "labelled_generic_names_V": bg["labelled"]["V"]["generic_names"],
        "pool_initialism_rates": {k: v.get("name is an initialism") for k, v in fe["pool"].items()}}

    # ------------------------------------------------------------ 7. test-matched labelled subset (non-France)
    ids1 = np.asarray(ctx.s[1].col("id"))
    tm = {"US": c1t == CTRY["US"], "India": c1t == CTRY["India"]}
    nonfr = tm["US"] | tm["India"]
    cells_test = np.zeros(n1, dtype=np.int64) - 1
    for i, c in enumerate(("US", "India")):
        cells_test[tm[c]] = i * 7 + npb(npred_test["Step3"][tm[c]])
    test_share = np.bincount(cells_test[nonfr], minlength=14) / nonfr.sum()
    cells_val = np.where(cv == CTRY["US"], 0, 7) + npb(npred_env["T"])
    val_cnt = np.bincount(cells_val, minlength=14)
    ok = test_share > 0
    N = int(np.floor(np.min(val_cnt[ok] / test_share[ok])))
    h = seeded_hash([f"S1-{int(x)}" for x in ids1[qv]], 20260925 + 1234)
    sub = np.zeros(nq, dtype=bool)
    for cell in np.flatnonzero(ok):
        members = np.flatnonzero(cells_val == cell)
        take = int(round(N * test_share[cell]))
        sub[members[np.argsort(h[members], kind="stable")[:take]]] = True
    (OUT / "test_matched_subset_s1_ids.txt").write_text(
        "\n".join(f"S1-{int(x)}" for x in ids1[qv[sub]]) + "\n", encoding="utf-8")
    # evaluate existing baseline predictions (T environment; and with S-sized blocking losses)
    cT = np.load(TD / "cand_T.npy")
    pT = np.load(TD / "p_T_A.npy")
    predT = cT[M.select(cT, pT, 0.69, True)]
    cS = np.load(SD / "cand_stress.npy")
    missed_S = true[~np.isin(true, cS, assume_unique=True)]
    hyb = predT[~np.isin(predT, missed_S, assume_unique=True)]
    predV = np.load(MD / "cand_val.npy")
    predV = predV[M.select(predV, np.load(TD / "p_V_A.npy"), 0.69, True)]
    ev = {}
    for tag, pr in (("V", predV), ("T", predT), ("T + S-sized blocking losses", hyb)):
        f, parts = M.entity_f05(pr, true, nq, return_parts=True)
        F = parts["F"]
        w_ctry = np.where(cv == CTRY["US"], tm["US"].sum() / nonfr.sum() / (cv == CTRY["US"]).mean(),
                          tm["India"].sum() / nonfr.sum() / (cv == CTRY["India"]).mean())
        wcell = test_share[cells_val] / np.maximum(val_cnt[cells_val] / nq, 1e-12)
        ev[tag] = {"all_validation_s1": float(F.mean()),
                   "country_mix_reweighted": float((w_ctry * F).sum() / w_ctry.sum()),
                   "country_x_predcount_reweighted": float((wcell * F).sum() / wcell.sum()),
                   "test_matched_subset": float(F[sub].mean()),
                   "subset_singleton_rate_true": float((ntrue[sub] == 0).mean()),
                   "subset_true_matches_per_s1": float(ntrue[sub].mean()),
                   "subset_by_predcount": {str(b): float(F[sub & (npb(npred_env['T']) == b)].mean())
                                           for b in NP_BUCKETS if (sub & (npb(npred_env['T']) == b)).any()}}
    val_share = val_cnt / nq
    res["test_matched_subset"] = {"n_s1": int(sub.sum()), "cells": "country (US, India) x Step3 predicted-count bucket (0..6+) in T",
                                  "test_cell_shares": test_share.tolist(), "validation_cell_shares": val_share.tolist(),
                                  "subset_cell_shares": (np.bincount(cells_val[sub], minlength=14) / sub.sum()).tolist(),
                                  "evaluation": ev}
    log(f"  subset {sub.sum():,} S1; eval {ev}")

    ha = {f: sha256(ROOT / f) for f in watch}
    res["hashes_after"] = ha
    res["unchanged"] = ha == hb
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  unchanged {res['unchanged']}; {res['seconds']}s")


if __name__ == "__main__":
    main()
