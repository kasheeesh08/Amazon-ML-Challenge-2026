#!/usr/bin/env python3
"""Target-context / t_rank ablation analysis (READ-ONLY diagnostic).

Nothing is retrained and nothing is re-inferred. The variant models were already trained with the exact Step 3
protocol by the completed t_rank diagnostic (experiments/diagnostics/trank/: model_*.npz, thresholds in select.json,
probabilities p_<env>_<variant>.npy for every candidate pair). This script only reads those artifacts and adds the
analyses that diagnostic did not report:
  * entity precision / recall per variant and environment, validation-vs-stress gaps
  * t_rank / t_gap / t_ncand distributions: normal validation (V), test-ratio (T), stress (S) and the REAL unlabelled
    test candidate set (experiments/test_scale t_*_test.npy), with PSI against V
  * the same features among true positives / false positives / false negatives of the current model
  * prediction-probability distributions per variant and environment (test: Step 3 model, prob_test.npy)

Variants: A current Step 3 model | B no t_rank | B2 no t_rank, t_gap, t_ncand | D t_rank as within-target percentile
t_rank/max(t_ncand-1,1) (the "stable representation") | C t_rank capped at 1 (extra, already available).
All outputs go to experiments/diagnostics/target_context_ablation/.
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
from test_scale_check import psi, sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
TD = ROOT / "experiments" / "diagnostics" / "trank"
MD = ROOT / "experiments" / "matching"
SD = ROOT / "experiments" / "test_scale" / "stress"
TS = ROOT / "experiments" / "test_scale"
FD = ROOT / "experiments" / "final"
C = MF.COL
VARIANTS = ["A", "B", "B2", "D", "C"]
ENVS = {"V": (MD / "cand_val.npy", MD / "X_val.npy"), "T": (TD / "cand_T.npy", TD / "X_T.npy"),
        "S": (SD / "cand_stress.npy", TD / "X_S.npy")}
QS = [10, 25, 50, 75, 90, 95, 99]
PBINS = [0.0, 0.1, 0.3, 0.5, 0.69, 0.8, 0.9, 0.95, 0.99, 1.0001]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, QS)
    return {"n": int(len(x)), "mean": float(x.mean()), **{f"p{k}": float(v) for k, v in zip(QS, q)}}


def column(xpath, name, chunk=2_000_000):
    X = np.load(xpath, mmap_mode="r")
    out = np.empty(X.shape[0], dtype=np.float32)
    for a in range(0, X.shape[0], chunk):
        out[a:a + chunk] = X[a:a + chunk, C[name]]
    return out


def main():
    t0 = time.time()
    hb = {f: sha256(ROOT / f) for f in PROTECTED}
    hb_final = hash_tree(FD)
    (OUT / "hashes_before.json").write_text(json.dumps({"protected": hb, "experiments_final": hb_final}, indent=1),
                                            encoding="utf-8")
    log(f"  hashed {len(hb)} protected files + {len(hb_final)} files in experiments/final")

    from blocking_experiment import Context
    ctx = Context(ROOT / "experiments" / "blocking" / "features", ROOT / "experiments" / "validation_split",
                  ROOT / "experiments" / "_eda_cache")
    qv = np.flatnonzero(ctx.fold[1] == 1)
    nq = len(qv)
    true = ctx.truth(qv)
    ntrue = np.bincount(true >> B.Q_SHIFT, minlength=nq)
    sel = json.loads((TD / "select.json").read_text(encoding="utf-8"))
    is_target = {k: np.zeros(ctx.s[k].n, dtype=bool) for k in (2, 3)}
    for k in (2, 3):
        is_target[k][ctx.pair_trow[ctx.pair_src == k]] = True

    res = {"metrics": {}, "feature_distributions": {}, "feature_by_outcome": {}, "probability_distributions": {},
           "psi_vs_V": {}}
    feats_env = {}
    for env, (cp, xp) in ENVS.items():
        te = time.time()
        codes = np.load(cp)
        q, s, t = B.decode(codes)
        y = np.isin(codes, true, assume_unique=True)
        fit = np.zeros(len(codes), dtype=bool)
        tgt = np.zeros(len(codes), dtype=bool)
        for k in (2, 3):
            m = s == k
            fit[m] = ctx.fold[k][t[m]] == 0
            tgt[m] = is_target[k][t[m]]
        f = {"t_rank": column(xp, "t_rank"), "t_gap": column(xp, "t_gap"), "t_ncand": column(xp, "t_ncand")}
        feats_env[env] = f
        res["feature_distributions"][env] = {k: summary(v) for k, v in f.items()}
        res["feature_distributions"][env]["share_t_rank0"] = float((f["t_rank"] == 0).mean())
        res["feature_distributions"][env]["share_uncontested(t_ncand=1)"] = float((f["t_ncand"] == 1).mean())
        res["metrics"][env] = {}
        res["probability_distributions"][env] = {}
        for v in VARIANTS:
            p = np.load(TD / f"p_{env}_{v}.npy")
            th = sel[v]["threshold"]
            keep = M.select(codes, p, th, True)
            pred = codes[keep]
            _, parts = M.entity_f05(pred, true, nq, return_parts=True)
            P, R, F, npred = parts["P"], parts["R"], parts["F"], parts["npred"]
            tp = int((keep & y).sum())
            fpm = keep & ~y
            single = ntrue == 0
            res["metrics"][env][v] = {
                "threshold": th, "macro_f05": float(F.mean()),
                "pair_precision": tp / int(keep.sum()), "pair_recall": tp / len(true),
                "entity_precision": float(P[npred > 0].mean()), "entity_recall": float(R[ntrue > 0].mean()),
                "singleton_f05": float(F[single].mean()), "singleton_false_merge_rate": float((npred[single] > 0).mean()),
                "nonsingleton_f05": float(F[~single].mean()),
                "predictions": int(keep.sum()), "tp": tp, "fp": int(fpm.sum()), "fn": int(len(true) - tp),
                "fn_blocked_rejected": int((~keep & y).sum()), "fn_not_in_candidates": int(len(true) - y.sum()),
                "fp_on_singleton_s1": int((fpm & (ntrue[q] == 0)).sum()),
                "fp_validation_pool": int((fpm & ~fit).sum()),
                "fp_fit_target_orphan": int((fpm & fit & tgt).sum()),
                "fp_fit_distractor": int((fpm & fit & ~tgt).sum()),
                "fp_t_rank0_uncontested": int((fpm & (f["t_rank"] == 0) & (f["t_ncand"] == 1)).sum()),
                "fp_t_rank0": int((fpm & (f["t_rank"] == 0)).sum())}
            h = np.histogram(p, bins=PBINS)[0]
            res["probability_distributions"][env][v] = {
                "candidates_share_by_bin": {f"[{a:.2f},{min(b, 1):.2f})": float(c / len(p)) for a, b, c in
                                            zip(PBINS[:-1], PBINS[1:], h)},
                "mean_prob_tp": float(p[keep & y].mean()), "mean_prob_fp": float(p[fpm].mean()),
                "share_fp_prob_ge_0.95": float((p[fpm] >= 0.95).mean()),
                "share_tp_prob_ge_0.95": float((p[keep & y] >= 0.95).mean()),
                "mean_prob_fn_blocked": float(p[~keep & y].mean())}
            if v == "A":
                res["feature_by_outcome"][env] = {
                    grp: {k: summary(val[m]) | {"share_t_rank0": float((f["t_rank"][m] == 0).mean()),
                                                "share_uncontested": float((f["t_ncand"][m] == 1).mean())}
                          for k, val in f.items()}
                    for grp, m in (("TP", keep & y), ("FP", fpm), ("FN_blocked", ~keep & y), ("TN", ~keep & ~y))}
            r = res["metrics"][env][v]
            log(f"  {env}/{v}: F0.5 {r['macro_f05']:.5f} pairP {r['pair_precision']:.4f} pairR {r['pair_recall']:.4f} "
                f"entP {r['entity_precision']:.4f} entR {r['entity_recall']:.4f} single {r['singleton_f05']:.4f} "
                f"FP {r['fp']:,} FN {r['fn']:,}")
        log(f"  env {env} done ({time.time() - te:.0f}s)")

    # real unlabelled test candidate set (label-free features + Step 3-model probabilities)
    tf = {k: np.load(TS / f"{k}_test.npy", mmap_mode="r") for k in ("t_rank", "t_gap", "t_ncand")}
    step = 5
    tsub = {k: np.asarray(v[::step]) for k, v in tf.items()}
    res["feature_distributions"]["TEST"] = {k: summary(v) for k, v in tsub.items()}
    res["feature_distributions"]["TEST"]["share_t_rank0"] = float((tsub["t_rank"] == 0).mean())
    res["feature_distributions"]["TEST"]["share_uncontested(t_ncand=1)"] = float((tsub["t_ncand"] == 1).mean())
    res["feature_distributions"]["TEST"]["note"] = f"every {step}th of 126,274,337 test candidate pairs"
    cand = np.load(TS / "cand_test.npy")
    p3 = np.load(TS / "prob_test.npy")
    keep3 = M.select(cand, p3, sel["A"]["threshold"], True)
    h = np.histogram(p3, bins=PBINS)[0]
    res["probability_distributions"]["TEST"] = {"A (Step 3 model, unlabelled)": {
        "candidates_share_by_bin": {f"[{a:.2f},{min(b, 1):.2f})": float(c / len(p3)) for a, b, c in
                                    zip(PBINS[:-1], PBINS[1:], h)},
        "predictions": int(keep3.sum()), "share_pred_prob_ge_0.95": float((p3[keep3] >= 0.95).mean())}}
    res["feature_by_outcome"]["TEST"] = {"predicted (Step 3 model)": {
        k: summary(np.asarray(tf[k])[keep3]) for k in tf} | {
        "share_t_rank0": float((np.asarray(tf["t_rank"])[keep3] == 0).mean()),
        "share_uncontested": float((np.asarray(tf["t_ncand"])[keep3] == 1).mean())}}
    del cand, p3, keep3
    rng = np.random.default_rng(20260925)
    for env in ("T", "S", "TEST"):
        res["psi_vs_V"][env] = {}
        for k in ("t_rank", "t_gap", "t_ncand"):
            a = feats_env["V"][k]
            b = tsub[k] if env == "TEST" else feats_env[env][k]
            ia = rng.choice(len(a), size=min(len(a), 2_000_000), replace=False)
            ib = rng.choice(len(b), size=min(len(b), 2_000_000), replace=False)
            res["psi_vs_V"][env][k] = float(psi(a[ia], b[ib]))
    log(f"  PSI vs V: {res['psi_vs_V']}")

    ha = {f: sha256(ROOT / f) for f in PROTECTED}
    ha_final = hash_tree(FD)
    res["integrity"] = {"protected_unchanged": ha == hb, "n_protected": len(hb),
                        "experiments_final_unchanged": ha_final == hb_final, "n_final_files": len(hb_final),
                        "output/matching_results.tsv": ha["output/matching_results.tsv"],
                        "output/candidate_pairs.tsv": ha["output/candidate_pairs.tsv"],
                        "experiments/final/model_final.npz": ha_final.get("experiments/final/model_final.npz")}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  integrity {res['integrity']}; total {res['seconds']}s")


if __name__ == "__main__":
    main()
