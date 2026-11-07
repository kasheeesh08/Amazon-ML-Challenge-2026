#!/usr/bin/env python3
"""Blocking / candidate-recall gap investigation (READ-ONLY diagnostic; nothing is re-blocked, retrained or re-inferred).

Uses only saved artifacts:
  config        experiments/blocking/blocking_config.json (+ hashes recorded by the final run)
  V  normal validation   experiments/matching/cand_val.npy, X_val.npy (nameaddr_cos column)
  T  test-ratio          experiments/diagnostics/trank/cand_T.npy, X_T.npy, poolT_s{2,3}.npy
  S  stress (full pool)  experiments/test_scale/stress/cand_stress.npy, experiments/diagnostics/trank/X_S.npy
  TEST (unlabelled)      experiments/test_scale/cand_test.npy, cos_test.npy, features_test/ (country, names, flags)
  channel diagnostics    experiments/blocking/checkpoints/validation_channels.json, trank/candidates_T.json,
                         test_scale/stress/candidates.json, test_scale/candidates.json (pair counts only)
  current-model FN split experiments/diagnostics/trank/evaluate.json
Outputs: experiments/diagnostics/blocking_gap/{results.json, run.log}; the report is written separately.
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
from ber import normalize as N  # noqa: E402
from ber.features import FeatureStore  # noqa: E402
from ber.normalize import h64  # noqa: E402
from test_scale_check import sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

OUT = Path(__file__).resolve().parent
MD = ROOT / "experiments" / "matching"
TD = ROOT / "experiments" / "diagnostics" / "trank"
SD = ROOT / "experiments" / "test_scale" / "stress"
TS = ROOT / "experiments" / "test_scale"
FD = ROOT / "experiments" / "final"
BD = ROOT / "experiments" / "blocking"
from ber.match_features import COL as _COL  # noqa: E402
COS_COL = _COL["nameaddr_cos"]
CTRY = {"US": h64("us"), "India": h64("india"), "France": h64("france")}
QS = [1, 5, 50, 90, 95, 99]
TFLAGS = {"target address empty": N.FLAG_ADDR_MISSING, "target name non-Latin": N.FLAG_NAME_NONLATIN,
          "target name domain-like": N.FLAG_NAME_DOMAIN, "target name junk prefix": N.FLAG_NAME_JUNK_PREFIX,
          "target address landmark": N.FLAG_ADDR_LANDMARK, "target postal present": N.FLAG_POSTAL_PRESENT}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, QS)
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
            **{f"p{k}": float(v) for k, v in zip(QS, q)}}


def count_block(cnt):
    return dist(cnt) | {"zero": int((cnt == 0).sum()), "exactly_1": int((cnt == 1).sum()),
                        "le_20": int((cnt <= 20).sum()), "ge_200": int((cnt >= 200).sum()),
                        "ge_300": int((cnt >= 300).sum()), "ge_400": int((cnt >= 400).sum()),
                        "ge_500": int((cnt >= 500).sum())}


def best_cos(xpath_or_arr, q, nq, is_arr=False, chunk=2_000_000):
    """Highest nameaddr_cos among each S1's candidates (codes are sorted by query, so reduceat over query starts)."""
    if is_arr:
        cos = np.asarray(xpath_or_arr, dtype=np.float32)
    else:
        X = np.load(xpath_or_arr, mmap_mode="r")
        cos = np.empty(X.shape[0], dtype=np.float32)
        for a in range(0, X.shape[0], chunk):
            cos[a:a + chunk] = X[a:a + chunk, COS_COL]
    starts = np.searchsorted(q, np.arange(nq))
    has = np.diff(np.append(starts, len(q))) > 0
    best = np.full(nq, -1.0, dtype=np.float32)
    best[has] = np.maximum.reduceat(cos, starts[has])
    return best


def main():
    t0 = time.time()
    hb = {f: sha256(ROOT / f) for f in PROTECTED}
    hb_final = hash_tree(FD)
    (OUT / "hashes_before.json").write_text(json.dumps({"protected": hb, "experiments_final": hb_final}, indent=1),
                                            encoding="utf-8")
    res = {}

    # ------------------------------------------------------------------ 1. configuration
    cfg_path = BD / "blocking_config.json"
    cfg = jload(cfg_path)
    cfg_hash = sha256(cfg_path)
    res["config"] = {"path": "experiments/blocking/blocking_config.json", "sha256": cfg_hash, "config": cfg,
                     "hash_recorded_by_final_run": jload(FD / "test_inputs.json")["blocking_config_sha256"],
                     "hash_in_submission_package": sha256(ROOT / "submission_package/code/business_entity_resolution/src/config/blocking_config.json")}
    res["config"]["all_hashes_equal"] = len({cfg_hash, res["config"]["hash_recorded_by_final_run"],
                                              res["config"]["hash_in_submission_package"]}) == 1

    # ------------------------------------------------------------------ labelled environments
    from blocking_experiment import Context
    ctx = Context(BD / "features", ROOT / "experiments" / "validation_split", ROOT / "experiments" / "_eda_cache")
    qv = np.flatnonzero(ctx.fold[1] == 1)
    nq = len(qv)
    c_v = np.asarray(ctx.s[1].col("country"))[qv]
    true = ctx.truth(qv)
    tq, ts, tt = B.decode(true)
    ntrue = np.bincount(tq, minlength=nq)
    # true-pair categories
    s1_house = np.asarray(ctx.s[1].col("house"))[qv[tq]]
    s1_n3 = np.asarray(ctx.s[1].col("name_n3"))[qv[tq]]
    t_flags = np.empty(len(true), dtype=np.int64)
    t_house = np.empty(len(true), dtype=np.int64)
    t_n3 = np.empty(len(true), dtype=np.int64)
    for k in (2, 3):
        m = ts == k
        t_flags[m] = np.asarray(ctx.s[k].col("flags"))[tt[m]]
        t_house[m] = np.asarray(ctx.s[k].col("house"))[tt[m]].astype(np.int64)
        t_n3[m] = np.asarray(ctx.s[k].col("name_n3"))[tt[m]].astype(np.int64)
    cats = {name: (t_flags & bit) > 0 for name, bit in TFLAGS.items()}
    cats["house number differs (both present)"] = (s1_house.astype(np.int64) != 0) & (t_house != 0) & (s1_house.astype(np.int64) != t_house)
    cats["sorted name differs (name_n3)"] = s1_n3.astype(np.int64) != t_n3
    mc = ntrue[tq]
    mbuckets = {"S1 with 1 true match": mc == 1, "S1 with 2-3": (mc >= 2) & (mc <= 3), "S1 with 4-5": (mc >= 4) & (mc <= 5),
                "S1 with 6+": mc >= 6}
    ev = jload(TD / "evaluate.json")
    names_v = np.asarray(ctx.s[1].col("name_n2"))[qv]
    name_freq_v = np.zeros(nq, dtype=np.int64)
    for c, h in CTRY.items():
        m = c_v == h
        if m.any():
            _, inv, cnt = np.unique(names_v[m], return_inverse=True, return_counts=True)
            name_freq_v[m] = cnt[inv]

    pools = {"V": {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)},
             "T": {k: np.load(TD / f"poolT_s{k}.npy") for k in (2, 3)},
             "S": {k: np.arange(ctx.s[k].n) for k in (2, 3)}}
    envs = {"V": (MD / "cand_val.npy", MD / "X_val.npy"), "T": (TD / "cand_T.npy", TD / "X_T.npy"),
            "S": (SD / "cand_stress.npy", TD / "X_S.npy")}
    res["labelled"] = {}
    for env, (cp, xp) in envs.items():
        te = time.time()
        codes = np.load(cp)
        q, s, _ = B.decode(codes)
        cnt = np.bincount(q, minlength=nq)
        cnt2 = np.bincount(q[s == 2], minlength=nq)
        cnt3 = np.bincount(q[s == 3], minlength=nq)
        hit = np.isin(true, codes, assume_unique=True)
        got = np.bincount(tq[hit], minlength=nq)
        pred_oracle = true[hit]
        _, parts = M.entity_f05(pred_oracle, true, nq, return_parts=True)
        bc = best_cos(xp, q, nq)
        r = {"candidate_pairs": int(len(codes)), "per_s1": count_block(cnt), "per_s1_s2": dist(cnt2), "per_s1_s3": dist(cnt3),
             "candidate_recall": float(hit.mean()),
             "recall_s2": float(hit[ts == 2].mean()), "recall_s3": float(hit[ts == 3].mean()),
             "matched_s1_all_true_retrieved": float((got[ntrue > 0] == ntrue[ntrue > 0]).mean()),
             "matched_s1_any_true_retrieved": float((got[ntrue > 0] > 0).mean()),
             "oracle_macro_f05_perfect_matcher_on_candidates": float(parts["F"].mean()),
             "true_pairs_missed": int((~hit).sum()),
             "current_model_fn_total": ev[env]["variants"]["A"]["fn"],
             "current_model_fn_not_in_candidates": ev[env]["variants"]["A"]["fn_not_in_candidates"],
             "by_country": {}, "by_match_count": {}, "by_target_category": {},
             "pool_s2_s3_per_s1": {}, "best_candidate_cos": {}, "generic_names": {}}
        pc = {k: np.asarray(ctx.s[k].col("country"))[pools[env][k]] for k in (2, 3)}
        for c, h in CTRY.items():
            m = c_v == h
            if not m.any():
                continue
            mt = m[tq]
            _, pp = M.entity_f05(pred_oracle[m[pred_oracle >> B.Q_SHIFT]], true[mt], nq, return_parts=True)
            r["by_country"][c] = {"n_s1": int(m.sum()), "per_s1": count_block(cnt[m]),
                                  "candidate_recall": float(hit[mt].mean()),
                                  "matched_s1_all_true_retrieved": float((got[m & (ntrue > 0)] == ntrue[m & (ntrue > 0)]).mean()),
                                  "oracle_macro_f05": float(pp["F"][m].mean())}
            r["pool_s2_s3_per_s1"][c] = float(sum(int((pc[k] == h).sum()) for k in (2, 3)) / m.sum())
            bm = bc[m]
            r["best_candidate_cos"][c] = dist(bm) | {"share_lt_0.3": float((bm < 0.3).mean()),
                                                     "share_lt_0.5": float((bm < 0.5).mean())}
            gen, uni = m & (name_freq_v >= 10), m & (name_freq_v == 1)
            r["generic_names"][c] = {"share_s1_name_shared_10+": float(gen.sum() / m.sum()),
                                     "cand_per_s1_generic10": float(cnt[gen].mean()) if gen.any() else None,
                                     "cand_per_s1_unique": float(cnt[uni].mean()),
                                     "recall_generic10": float(hit[gen[tq]].mean()) if gen.any() else None,
                                     "recall_unique": float(hit[uni[tq]].mean())}
        low = bc < 0.3
        r["best_cos_lt_0.3_calibration"] = {
            "s1": int(low.sum()), "share_singleton": float((ntrue[low] == 0).mean()),
            "share_with_a_true_match_missed_by_blocking": float(((ntrue - got)[low] > 0).mean()),
            "share_all_s1_with_missed_true": float(((ntrue - got) > 0).mean())}
        for name, m in mbuckets.items():
            r["by_match_count"][name] = {"true_pairs": int(m.sum()), "candidate_recall": float(hit[m].mean())}
        for name, m in cats.items():
            r["by_target_category"][name] = {"true_pairs": int(m.sum()), "share_of_true_pairs": float(m.mean()),
                                             "candidate_recall": float(hit[m].mean()) if m.any() else None,
                                             "missed": int((m & ~hit).sum()),
                                             "share_of_all_missed": float((m & ~hit).sum() / max(1, (~hit).sum()))}
        res["labelled"][env] = r
        log(f"  {env}: recall {r['candidate_recall']:.5f}, oracle F0.5 {r['oracle_macro_f05_perfect_matcher_on_candidates']:.5f}, "
            f"cand/S1 {r['per_s1']['mean']:.1f} ({time.time() - te:.0f}s)")
        del codes, q, s, hit

    # ------------------------------------------------------------------ channels
    vch = jload(BD / "checkpoints" / "validation_channels.json")
    res["channels"] = {"V": vch, "T": jload(TD / "candidates_T.json")["channels"],
                       "S": jload(SD / "candidates.json")["channels"]}
    tc = jload(TS / "candidates.json")
    tot = {}
    for ch in tc["chunks"]:
        for k, v in ch["channels"].items():
            tot[k] = tot.get(k, 0) + v
    res["channels"]["TEST_pair_counts"] = tot

    # ------------------------------------------------------------------ real test (unlabelled)
    S1 = FeatureStore(TS / "features_test", "test_s1")
    tp = {k: FeatureStore(TS / "features_test", f"test_s{k}") for k in (2, 3)}
    n1 = S1.n
    c1 = np.asarray(S1.col("country"))
    cand = np.load(TS / "cand_test.npy")
    q, s, _ = B.decode(cand)
    cnt = np.bincount(q, minlength=n1)
    cnt2 = np.bincount(q[s == 2], minlength=n1)
    cnt3 = np.bincount(q[s == 3], minlength=n1)
    bc = best_cos(np.load(TS / "cos_test.npy"), q, n1, is_arr=True)
    names = np.asarray(S1.col("name_n2"))
    rt = {"candidate_pairs": int(len(cand)), "per_s1": count_block(cnt), "per_s1_s2": dist(cnt2), "per_s1_s3": dist(cnt3),
          "by_country": {}}
    pcty = {k: np.asarray(tp[k].col("country")) for k in (2, 3)}
    pflags = {k: np.asarray(tp[k].col("flags")).astype(np.int64) for k in (2, 3)}
    for c, h in CTRY.items():
        m = c1 == h
        _, inv, fcnt = np.unique(names[m], return_inverse=True, return_counts=True)
        freq = fcnt[inv]
        cm = cnt[m]
        bm = bc[m]
        pool_n = sum(int((pcty[k] == h).sum()) for k in (2, 3))
        pool_rates = {}
        for name, bit in TFLAGS.items():
            num = sum(int(((pflags[k] & bit) > 0)[pcty[k] == h].sum()) for k in (2, 3))
            pool_rates[name.replace("target ", "pool ")] = num / pool_n
        rt["by_country"][c] = {
            "n_s1": int(m.sum()), "per_s1": count_block(cm), "per_s1_s2": dist(cnt2[m]), "per_s1_s3": dist(cnt3[m]),
            "pool_s2_s3_per_s1": pool_n / int(m.sum()),
            "best_candidate_cos": dist(bm) | {"share_lt_0.3": float((bm < 0.3).mean()),
                                              "share_lt_0.5": float((bm < 0.5).mean())},
            "generic_names": {"share_s1_name_shared_10+": float((freq >= 10).mean()),
                              "share_s1_name_shared_100+": float((freq >= 100).mean()),
                              "cand_per_s1_generic10": float(cm[freq >= 10].mean()),
                              "cand_per_s1_generic100": float(cm[freq >= 100].mean()) if (freq >= 100).any() else None,
                              "cand_per_s1_unique": float(cm[freq == 1].mean())},
            "pool_flag_rates": pool_rates}
        log(f"  TEST {c}: cand/S1 {cm.mean():.1f}, best cos<0.3 {(bm < 0.3).mean():.4f}, pool/S1 {pool_n / m.sum():.3f}")
    # training-pool flag rates by country, for comparison
    rt["train_pool_flag_rates"] = {}
    for c, h in CTRY.items():
        pc = {k: np.asarray(ctx.s[k].col("country")) for k in (2, 3)}
        n = sum(int((pc[k] == h).sum()) for k in (2, 3))
        if n == 0:
            continue
        fl = {k: np.asarray(ctx.s[k].col("flags")).astype(np.int64) for k in (2, 3)}
        rt["train_pool_flag_rates"][c] = {name.replace("target ", "pool "): sum(
            int(((fl[k] & bit) > 0)[pc[k] == h].sum()) for k in (2, 3)) / n for name, bit in TFLAGS.items()}
    res["test"] = rt

    ha = {f: sha256(ROOT / f) for f in PROTECTED}
    ha_final = hash_tree(FD)
    res["integrity"] = {"protected_unchanged": ha == hb, "n_protected": len(hb),
                        "experiments_final_unchanged": ha_final == hb_final, "n_final_files": len(hb_final),
                        "output/matching_results.tsv": ha["output/matching_results.tsv"],
                        "output/candidate_pairs.tsv": ha["output/candidate_pairs.tsv"],
                        "blocking_config": ha["experiments/blocking/blocking_config.json"]}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  integrity {res['integrity']}; total {res['seconds']}s")


if __name__ == "__main__":
    main()
