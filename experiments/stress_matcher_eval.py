#!/usr/bin/env python3
"""Labelled test-scale stress evaluation of the EXISTING Step 3 matcher (evaluation only).

Environment B ("stress"): the SAME 441,364 validation S1 as Step 3, matched against the FULL training S2/S3
pool (10,320,219 records — test-sized). The 8.26M fit-fold records added to the validation pool are true
non-matches for validation S1 (verified in Step 1), so the validation ground truth remains complete.
Environment A is the saved Step 3 validation run (experiments/matching/*).

Unchanged components: Step 2 blocking config, Step 3 features (IDF from the pool being matched — here the
full training pool, as the Step 3 design prescribes), saved GBDT (300 trees), threshold 0.69, one-per-target
rule — all read from experiments/matching/selection.json. Nothing is tuned. No submission is written.

Phases (checkpointed in experiments/test_scale/stress/): candidates, idf, cos, target_context, infer, evaluate,
errors, report
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.ids_tsv import MATCHING_HEADER  # noqa: E402
from test_scale_check import (SHIFT_FEATURES, chunks_by_query, dist, jdump, jload, peak_rss, psi,  # noqa: E402
                              rank_buckets, sample_mask, sha256)

N_TARGET_PARTS = 8
PHASES = ["candidates", "idf", "cos", "target_context", "infer", "evaluate", "errors", "report"]


def log(*a):
    print(*a, flush=True)


def mem():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_working_set_gb": round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2),
                "private_gb_now": round(getattr(mi, "private", mi.rss) / 1e9, 2),
                "system_free_gb_now": round(psutil.virtual_memory().available / 1e9, 2)}
    except Exception:
        return {}


class Stress:
    def __init__(self, a):
        from blocking_experiment import Context
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.md = Path(a.matching_dir)
        self.ctx = Context(a.feat_dir, a.split_dir, a.cache_dir)
        self.qv = np.flatnonzero(self.ctx.fold[1] == 1)
        self.pool = {k: np.arange(self.ctx.s[k].n) for k in (2, 3)}
        sel = jload(self.md / "selection.json")["chosen"]
        self.spec = {k: sel[k] for k in ("model", "n_trees", "one_to_one", "best_threshold")}
        self.config = jload(a.blocking_config)
        self.stores = {2: self.ctx.s[2], 3: self.ctx.s[3]}

    def p(self, n):
        return self.out / n

    # ----------------------------------------------------------------------------------------
    def candidates(self):
        from blocking_experiment import lean_union
        true = self.ctx.truth(self.qv)
        union, info = lean_union(self.ctx, self.config, self.qv, self.pool, true)
        np.save(self.p("cand_stress.npy"), union)
        jdump(self.p("candidates.json"), {"pairs": int(len(union)), "channels": info,
                                          "candidate_recall": float(np.isin(true, union, assume_unique=True).mean()),
                                          "true_pairs": int(len(true)), "memory": mem()})
        log(f"  stress candidates {len(union):,}; candidate recall {np.isin(true, union, assume_unique=True).mean():.6f}")

    def idf(self):
        for ch in ("name", "addr"):
            MF.build_idf(self.stores, self.pool, ch).save(self.p(f"idf_full_{ch}.npz"))

    def _X(self, codes, idf_n, idf_a):
        return MF.pair_features(self.ctx.s[1], self.qv, self.stores, codes, idf_n, idf_a)

    def cos(self):
        codes = np.load(self.p("cand_stress.npy"), mmap_mode="r")
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_full_{c}.npz")) for c in ("name", "addr"))
        cos = np.lib.format.open_memmap(self.p("cos_stress.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        for a, b in chunks_by_query(np.asarray(codes)):
            cos[a:b] = self._X(np.asarray(codes[a:b]), idf_n, idf_a)[:, MF.COL["nameaddr_cos"]]
        cos.flush()

    def target_context(self):
        """Per target-row partition (a target's group never spans partitions) — identical to MF.target_context."""
        codes = np.load(self.p("cand_stress.npy"))
        cos = np.load(self.p("cos_stress.npy"), mmap_mode="r")
        _, s, t = B.decode(codes)
        P = N_TARGET_PARTS
        pid = (s - 2) * P + np.minimum((t * P) // np.where(s == 2, self.ctx.s[2].n, self.ctx.s[3].n), P - 1)
        order = np.argsort(pid, kind="stable")
        bnd = np.searchsorted(pid[order], np.arange(2 * P + 1))
        del s, t
        outs = {nm: np.lib.format.open_memmap(self.p(f"{nm}_stress.npy"), mode="w+", dtype=np.float32,
                                              shape=(len(codes),)) for nm in ("t_rank", "t_gap", "t_ncand")}
        for k in range(2 * P):
            sel = np.sort(order[bnd[k]:bnd[k + 1]])
            if len(sel):
                r, g, n = MF.target_context(codes[sel], np.asarray(cos[sel]))
                outs["t_rank"][sel], outs["t_gap"][sel], outs["t_ncand"][sel] = r, g, n
        for o in outs.values():
            o.flush()

    def infer(self):
        codes = np.load(self.p("cand_stress.npy"), mmap_mode="r")
        tc = {nm: np.load(self.p(f"{nm}_stress.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_full_{c}.npz")) for c in ("name", "addr"))
        gb = GBDT.load(self.md / "model_gbdt.npz")
        prob = np.lib.format.open_memmap(self.p("prob_stress.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        fsum = np.zeros(MF.N_FEATURES)
        samples = []
        t0 = time.time()
        for a, b in chunks_by_query(np.asarray(codes)):
            c = np.asarray(codes[a:b])
            X = self._X(c, idf_n, idf_a)
            for nm in tc:
                X[:, MF.COL[nm]] = tc[nm][a:b]
            prob[a:b] = gb.predict_proba(X, n_trees=self.spec["n_trees"])
            fsum += X.sum(axis=0, dtype=np.float64)
            samples.append(X[sample_mask(c)])
        prob.flush()
        np.save(self.p("feature_sample_stress.npy"), np.concatenate(samples))
        jdump(self.p("infer.json"), {"pairs_scored": int(len(codes)), "seconds": round(time.time() - t0, 1),
                                     "feature_means": dict(zip(MF.FEATURE_NAMES, (fsum / len(codes)).tolist())),
                                     "model_sha256": sha256(self.md / "model_gbdt.npz"), "spec": self.spec,
                                     "memory": mem()})

    # ----------------------------------------------------------------------------------------
    def _metrics(self, codes, prob, true, n_q):
        th, one = self.spec["best_threshold"], self.spec["one_to_one"]
        pre = np.flatnonzero(prob >= th)
        c_pre, p_pre = codes[pre], prob[pre]
        q, s, t = B.decode(c_pre)
        _, tcnt = np.unique((s.astype(np.int64) << 32) | t, return_counts=True)
        keep = M.select(c_pre, p_pre, th, one)
        pred = c_pre[keep]
        blocked = true[np.isin(true, codes, assume_unique=True)]
        off, parts = M.entity_f05(pred, true, n_q, return_parts=True)
        nt, npd = parts["ntrue"], parts["npred"]
        single = nt == 0
        # accuracy of the rule's removals
        removed = c_pre[~keep]
        res = {"official_macro_f05": off, "pair": M.pair_metrics(pred, true, blocked),
               "entity_precision_mean": float(parts["P"][npd > 0].mean()),
               "entity_recall_mean": float(parts["R"][nt > 0].mean()),
               "f05_singletons": float(parts["F"][single].mean()), "f05_nonsingletons": float(parts["F"][~single].mean()),
               "n_singletons": int(single.sum()), "singleton_false_merge_rate": float((npd[single] > 0).mean()),
               "nonsingleton_empty_prediction_rate": float((npd[~single] == 0).mean()),
               "exact_set_match_rate": float(((parts["tp"] == nt) & (npd == nt)).mean()),
               "candidate_recall_ceiling": float(len(blocked) / len(true)),
               "rule": {"pred_before": int(len(pre)), "pred_after": int(len(pred)), "removed": int(len(removed)),
                        "removed_that_were_true_matches": int(np.isin(removed, true).sum()),
                        "targets_with_multiple_s1_before_rule": int((tcnt > 1).sum())},
               "by_source": {}}
        _, ps, _ = B.decode(pred)
        _, ts, _ = B.decode(true)
        for k in (2, 3):
            res["by_source"][f"S{k}"] = M.pair_metrics(pred[ps == k], true[ts == k], blocked[B.decode(blocked)[1] == k])
        ctry = np.asarray(self.ctx.s[1].col("country"))[self.qv]
        from ber.normalize import h64
        res["by_country"] = {c: {"n": int((ctry == h64(c.casefold())).sum()),
                                 "macro_f05": float(parts["F"][ctry == h64(c.casefold())].mean())}
                             for c in ("US", "India", "France") if (ctry == h64(c.casefold())).any()}
        return res, pred, keep, pre

    def evaluate(self):
        codes = np.load(self.p("cand_stress.npy"))
        prob = np.load(self.p("prob_stress.npy"))
        true = self.ctx.truth(self.qv)
        res, pred, _, _ = self._metrics(codes, prob, true, len(self.qv))
        per_q = np.bincount(codes >> B.Q_SHIFT, minlength=len(self.qv))
        res["candidates_per_s1"] = dist(per_q) | {"zero": int((per_q == 0).sum())}
        tr = np.load(self.p("t_rank_stress.npy"))
        tn = np.load(self.p("t_ncand_stress.npy"))
        res["t_rank"] = dist(tr) | {"buckets": rank_buckets(tr)}
        res["t_ncand"] = dist(tn)
        res["pairs"] = int(len(codes))
        res["pool"] = {"S2": int(self.ctx.s[2].n), "S3": int(self.ctx.s[3].n)}
        # Step 1 scorer cross-check on a validation-only prediction file (not a submission)
        ids = {k: np.asarray(self.ctx.s[k].col("id")) for k in (1, 2, 3)}
        path = self.p("val_stress_predictions.tsv")
        M.write_id_lists(path, MATCHING_HEADER, pred, [f"S1-{int(x)}" for x in ids[1][self.qv]], {2: ids[2], 3: ids[3]})
        from score import score_files
        rep = score_files(path, Path(self.a.split_dir) / "val_ground_truth.tsv",
                          test_dir=str(Path(self.a.split_dir) / "val_as_test_dir"))
        res["step1_scorer_macro_f05"] = rep.macro_f05
        res["step1_scorer_matches_vectorised"] = abs(rep.macro_f05 - res["official_macro_f05"]) < 1e-12
        # which pool do the false positives come from? (fit-fold records are new competition in B)
        fp = pred[~np.isin(pred, true, assume_unique=True)]
        _, fs, ft = B.decode(fp)
        fold = np.empty(len(fp), dtype=np.int8)
        for k in (2, 3):
            fold[fs == k] = self.ctx.fold[k][ft[fs == k]]
        res["false_positive_target_fold"] = {"validation_pool": int((fold == 1).sum()), "fit_pool_added_in_stress": int((fold == 0).sum())}
        jdump(self.p("evaluate.json"), res)
        np.save(self.p("stress_pred_codes.npy"), pred)
        log(f"  STRESS official macro F0.5 = {res['official_macro_f05']:.6f} (Step 1 scorer {rep.macro_f05:.6f}); "
            f"pair P={res['pair']['precision']:.4f} R={res['pair']['recall']:.4f}")

    def errors(self):
        from matching_experiment import Run
        codes = np.load(self.p("cand_stress.npy"))
        prob = np.load(self.p("prob_stress.npy"))
        pred = np.load(self.p("stress_pred_codes.npy"))
        true = self.ctx.truth(self.qv)
        sel = np.isin(codes, pred, assume_unique=True)
        y = np.isin(codes, true, assume_unique=True)
        ntrue = np.bincount(true >> B.Q_SHIFT, minlength=len(self.qv))
        q = codes >> B.Q_SHIFT
        fp, tp, fn = np.flatnonzero(sel & ~y), np.flatnonzero(sel & y), np.flatnonzero(~sel & y)
        # features for the error rows: recompute from the stored components (chunk-wise, only needed rows)
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_full_{c}.npz")) for c in ("name", "addr"))
        tc = {nm: np.load(self.p(f"{nm}_stress.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        rng = np.random.default_rng(20260925)

        def feats(idx):
            idx = np.sort(idx)
            X = self._X(codes[idx], idf_n, idf_a)        # rank_src/all context not valid for a subset -> excluded below
            for nm in tc:
                X[:, MF.COL[nm]] = tc[nm][idx]
            return X, idx
        n_s = 200_000

        def profile(idx):
            if len(idx) == 0:
                return {}
            sub = np.sort(rng.choice(idx, size=min(n_s, len(idx)), replace=False))
            X, sub = feats(sub)
            f = lambda c: X[:, MF.COL[c]]  # noqa: E731
            return {"n": int(len(idx)), "profiled": int(len(sub)),
                    "same_name_different_address(name_cos>=0.8 & addr_cos<0.3)": float(((f("name_cos") >= 0.8) & (f("addr_cos") < 0.3)).mean()),
                    "same_address_different_name(addr_cos>=0.8 & name_cos<0.3)": float(((f("addr_cos") >= 0.8) & (f("name_cos") < 0.3)).mean()),
                    "legal_form_or_punctuation_only(eq_name_n2 & !eq_name_n1)": float(((f("eq_name_n2") == 1) & (f("eq_name_n1") == 0)).mean()),
                    "target_address_empty": float(f("t_addr_missing").mean()),
                    "target_name_non_latin": float(f("t_name_nonlatin").mean()),
                    "house_numbers_differ": float(((f("house_both") == 1) & (f("eq_house") == 0)).mean()),
                    "target_competed_for_by_2+_S1(t_ncand>=2)": float((f("t_ncand") >= 2).mean()),
                    "pair_not_best_S1_for_target(t_rank>0)": float((f("t_rank") > 0).mean()),
                    "query_is_singleton": float((ntrue[q[sub]] == 0).mean())}
        E = {"counts": {"TP": int(len(tp)), "FP": int(len(fp)), "FN_blocked_rejected": int(len(fn)),
                        "FN_not_in_candidates": int((~np.isin(true, codes, assume_unique=True)).sum()),
                        "FP_on_singleton_S1": int((ntrue[q[fp]] == 0).sum())},
             "profile": {"TP": profile(tp), "FP": profile(fp), "FN_blocked_rejected": profile(fn)}}
        runner = SimpleNamespace(args=SimpleNamespace(data_dir=self.a.data_dir))
        pick = {
            "top_confidence_false_positives": fp[np.argsort(-prob[fp], kind="stable")[:10]],
            "singleton_false_merges": fp[ntrue[q[fp]] == 0][np.argsort(-prob[fp][ntrue[q[fp]] == 0], kind="stable")[:8]],
            "false_negatives_blocked_but_rejected": np.sort(rng.choice(fn, size=min(10, len(fn)), replace=False)),
        }
        E["examples"] = {k: Run._examples(runner, codes[v], prob[v], self.qv) for k, v in pick.items()}
        jdump(self.p("errors.json"), E)
        log(f"  errors: {E['counts']}")

    def report(self):
        write_report(self)


def pct(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def write_report(st):
    out = st.out
    A = jload(st.md / "val_eval.json")["chosen"]
    Aref = jload(Path(st.a.test_scale_dir) / "val_reference.json")
    Bm, C, I, Er = (jload(out / n) for n in ("evaluate.json", "candidates.json", "infer.json", "errors.json"))
    T = jload(out / "timings.json")
    Fa = np.load(Path(st.a.test_scale_dir) / "feature_sample_val.npy")
    Fb = np.load(out / "feature_sample_stress.npy")
    shift = {f: {"A_mean": Aref["feature_means"][f], "B_mean": I["feature_means"][f],
                 "A_p50": float(np.median(Fa[:, MF.COL[f]])), "B_p50": float(np.median(Fb[:, MF.COL[f]])),
                 "psi": psi(Fa[:, MF.COL[f]], Fb[:, MF.COL[f]])} for f in SHIFT_FEATURES}
    jdump(out / "shift.json", shift)
    hb = jload(ROOT / "experiments" / "test_scale" / "stress" / "hashes_before.json")
    ha = {f: sha256(ROOT / f) for f in hb}
    unchanged = all(ha[f] == hb[f] for f in hb)
    jdump(out / "hashes_after.json", ha)
    L = []
    w = L.append
    w("# Labelled test-scale stress evaluation of the Step 3 matcher\n")
    w("Generated by `experiments/stress_matcher_eval.py`. Evaluation only: nothing retrained, tuned or changed; no "
      "matching_results.tsv or submission. (The earlier unlabelled test-set diagnostic is `experiments/test_scale/TEST_SCALE_REPORT.md`.)\n")
    w("## A. Configuration\n")
    w(f"- Matcher: `experiments/matching/model_gbdt.npz` (SHA-256 `{I['model_sha256']}`), spec from `selection.json`: `{I['spec']}`.")
    w("- Blocking: `experiments/blocking/blocking_config.json` (unchanged); features: `src/ber/match_features.py` (unchanged), "
      "IDF from the pool being matched (Step 3 design) — here the full training pool.")
    w("- **A (normal)** = saved Step 3 run: validation S1 → validation pool. **B (stress)** = the same validation S1 → the FULL training "
      "S2/S3 pool. The added fit-fold records are true non-matches for validation S1, so labels stay complete. No test data or labels; no France.\n")
    w("## B–C. Sizes and candidates\n")
    w("| | A: validation pool | B: full training pool | change |")
    w("|---|---:|---:|---:|")
    w(f"| S1 queries | {Aref['n_s1']:,} | {len(st.qv):,} | — |")
    pa, pb = Aref['pool']['S2'] + Aref['pool']['S3'], Bm['pool']['S2'] + Bm['pool']['S3']
    w(f"| pool S2+S3 | {pa:,} | {pb:,} | ×{pb / pa:.2f} |")
    w(f"| S1 per pool record | {Aref['n_s1'] / pa:.4f} | {len(st.qv) / pb:.4f} | (test: 0.1738) |")
    w(f"| candidate pairs | {Aref['pairs']:,} | {Bm['pairs']:,} | {100 * (Bm['pairs'] / Aref['pairs'] - 1):+.1f}% |")
    w(f"| candidate recall (ceiling) | {pct(A['candidate_recall_ceiling'], 3)} | {pct(Bm['candidate_recall_ceiling'], 3)} | "
      f"{100 * (Bm['candidate_recall_ceiling'] - A['candidate_recall_ceiling']):+.3f} pp |")
    for k in ("mean", "p50", "p90", "p95", "p99", "max"):
        w(f"| candidates/S1 {k} | {Aref['candidates_per_s1'][k]:.1f} | {Bm['candidates_per_s1'][k]:.1f} | "
          f"{Bm['candidates_per_s1'][k] - Aref['candidates_per_s1'][k]:+.1f} |")
    w(f"| targets claimed by 2+ S1 before rule | {Aref['targets_with_multiple_s1_before_rule']:,} | "
      f"{Bm['rule']['targets_with_multiple_s1_before_rule']:,} | ×{Bm['rule']['targets_with_multiple_s1_before_rule'] / max(1, Aref['targets_with_multiple_s1_before_rule']):.1f} |")
    w("")
    w("## D. Metrics (official metric = Step 1 definition; B cross-checked with `src/score.py`)\n")
    w("| metric | A normal | B stress | abs. change | rel. change |")
    w("|---|---:|---:|---:|---:|")

    def row(lab, a, b, is_pct=True, d=4):
        if is_pct:
            w(f"| {lab} | {pct(a)} | {pct(b)} | {100 * (b - a):+.2f} pp | {100 * (b / a - 1) if a else 0:+.2f}% |")
        else:
            w(f"| {lab} | {a:.{d}f} | {b:.{d}f} | {b - a:+.{d}f} | {100 * (b / a - 1) if a else 0:+.2f}% |")
    row("**official macro F0.5**", A["official_macro_f05"], Bm["official_macro_f05"], False, 6)
    for lab, k in (("pair precision", "precision"), ("pair recall (end-to-end)", "recall"),
                   ("pair recall among blocked", "recall_among_blocked"), ("pair F0.5", "f05"), ("pair F1", "f1")):
        row(lab, A["pair"][k], Bm["pair"][k])
    for lab, k in (("entity precision", "entity_precision_mean"), ("entity recall", "entity_recall_mean"),
                   ("F0.5 singletons", "f05_singletons"), ("singleton false-merge rate", "singleton_false_merge_rate"),
                   ("F0.5 S1 with matches", "f05_nonsingletons"),
                   ("S1 with matches but empty prediction", "nonsingleton_empty_prediction_rate"),
                   ("S1 predicted exactly right", "exact_set_match_rate")):
        row(lab, A[k], Bm[k])
    w(f"| predicted pairs | {A['pair']['n_pred']:,} | {Bm['pair']['n_pred']:,} | {Bm['pair']['n_pred'] - A['pair']['n_pred']:+,} | "
      f"{100 * (Bm['pair']['n_pred'] / A['pair']['n_pred'] - 1):+.2f}% |")
    w(f"| true positives | {A['pair']['tp']:,} | {Bm['pair']['tp']:,} | {Bm['pair']['tp'] - A['pair']['tp']:+,} | "
      f"{100 * (Bm['pair']['tp'] / A['pair']['tp'] - 1):+.2f}% |")
    for s in ("S2", "S3"):
        row(f"{s} pair precision", A["by_source"][s]["pair"]["precision"], Bm["by_source"][s]["precision"])
        row(f"{s} pair recall", A["by_source"][s]["pair"]["recall"], Bm["by_source"][s]["recall"])
    w(f"\nBy country (B): " + ", ".join(f"{c} {v['macro_f05']:.4f} (n={v['n']:,})" for c, v in Bm["by_country"].items())
      + "; A: India 0.9503, US 0.9712.")
    w(f"\nStep 1 scorer on the B prediction file: {Bm['step1_scorer_macro_f05']:.6f} — equal to the vectorised value: "
      f"{Bm['step1_scorer_matches_vectorised']}.\n")
    r = Bm["rule"]
    w(f"One-per-target rule (B): {r['pred_before']:,} predicted before, {r['removed']:,} removed "
      f"({100 * r['removed'] / r['pred_before']:.2f}%), of which {r['removed_that_were_true_matches']:,} were true matches; "
      f"A: {Aref['pred_before_rule']:,} before, {Aref['removed_by_one_to_one_rule']:,} removed.\n")
    fpf = Bm["false_positive_target_fold"]
    w(f"False positives in B by origin of the target record: validation pool {fpf['validation_pool']:,}; fit-fold records added "
      f"in the stress pool {fpf['fit_pool_added_in_stress']:,}.\n")
    w("## E. t_rank and ranking features\n")
    w("| | A | B |")
    w("|---|---:|---:|")
    for k in ("mean", "p50", "p75", "p90", "p95", "p99", "max"):
        w(f"| t_rank {k} | {Aref['t_rank'][k]:.2f} | {Bm['t_rank'][k]:.2f} |")
    for bk in Aref["t_rank"]["buckets"]:
        w(f"| share t_rank = {bk} | {pct(Aref['t_rank']['buckets'][bk])} | {pct(Bm['t_rank']['buckets'][bk])} |")
    w(f"| t_ncand mean / p50 / p90 | — | {Bm['t_ncand']['mean']:.1f} / {Bm['t_ncand']['p50']:.0f} / {Bm['t_ncand']['p90']:.0f} |")
    w("\n| feature | A mean | B mean | A p50 | B p50 | PSI |")
    w("|---|---:|---:|---:|---:|---:|")
    for f, d in sorted(shift.items(), key=lambda kv: -kv[1]["psi"]):
        flag = " **major**" if d["psi"] > 0.25 else (" moderate" if d["psi"] > 0.1 else "")
        w(f"| `{f}` | {d['A_mean']:.4f} | {d['B_mean']:.4f} | {d['A_p50']:.3f} | {d['B_p50']:.3f} | {d['psi']:.4f}{flag} |")
    w("\n## F. Singleton behaviour\n")
    w(f"- Singletons: {Bm['n_singletons']:,}. F0.5 {pct(A['f05_singletons'])} → {pct(Bm['f05_singletons'])}; false-merge rate "
      f"{pct(A['singleton_false_merge_rate'])} → {pct(Bm['singleton_false_merge_rate'])}; FPs on singleton S1: "
      f"{Er['counts']['FP_on_singleton_S1']:,} in B.\n")
    w("## G. Error analysis (B)\n")
    c = Er["counts"]
    w(f"TP {c['TP']:,}; FP {c['FP']:,}; FN blocked-but-rejected {c['FN_blocked_rejected']:,}; FN not in candidates {c['FN_not_in_candidates']:,}.\n")
    keys = [k for k in Er["profile"]["TP"] if k not in ("n", "profiled")]
    w("| share with property (random sample ≤200k per group) | TP | FP | FN (blocked, rejected) |")
    w("|---|---:|---:|---:|")
    for k in keys:
        w(f"| {k} | " + " | ".join(pct(Er["profile"][g].get(k)) for g in ("TP", "FP", "FN_blocked_rejected")) + " |")
    w("")
    for k, lst in Er["examples"].items():
        w(f"### {k.replace('_', ' ')}\n")
        for e in lst:
            s1, t = e["s1"], e["candidate"]
            w(f"- p={e['prob']:.3f}: **{s1[1]}** | {s1[2]} ({s1[3]})  ⟷  `{t[0]}` **{t[1]}** | {t[2] or '∅'} ({t[3]}) — {', '.join(e['differences'])}")
        w("")
    w("## H. Memory / runtime\n")
    w(f"- Phase times (s): {T}")
    w(f"- Memory: candidates {C.get('memory')}; inference {I.get('memory')} (peak working set includes memory-mapped file pages).")
    w("- Streaming: candidates in one lean-union pass (channel by channel), features recomputed chunk-wise (2M pairs), target context "
      "per target-row partition; no full feature matrix stored.\n")
    w("## I. Stability — see the assistant's summary for the interpretation.\n")
    w("## J–K. Integrity\n")
    w(f"- All {len(hb)} Step 1–3 source/config/model files hash-identical before vs after: **{unchanged}** "
      "(`hashes_before.json`, `hashes_after.json`).")
    w("- France: no labelled France records exist (train = US + India), so this experiment gives no evidence about France.")
    Path(st.a.report_path).write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {st.a.report_path} (hashes unchanged: {unchanged})")


PHASE_OUT = {"candidates": "candidates.json", "idf": "idf_full_addr.npz", "cos": "cos_stress.npy",
             "target_context": "t_ncand_stress.npy", "infer": "infer.json", "evaluate": "evaluate.json",
             "errors": "errors.json", "report": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "dataset" / "train"))
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--blocking-config", default=str(ROOT / "experiments" / "blocking" / "blocking_config.json"))
    ap.add_argument("--matching-dir", default=str(ROOT / "experiments" / "matching"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "test_scale" / "stress"))
    ap.add_argument("--test-scale-dir", default=str(ROOT / "experiments" / "test_scale"),
                    help="where val_reference.json / feature_sample_val.npy of environment A live")
    ap.add_argument("--report-path", default=str(ROOT / "experiments" / "TEST_SCALE_REPORT.md"))
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    a = ap.parse_args()
    st = Stress(a)
    tp = st.p("timings.json")
    timings = jload(tp) if tp.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and st.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(st, ph)()
        timings[ph] = round(time.time() - t0, 1)
        jdump(tp, timings)
        log(f"[{ph}] done ({timings[ph]}s) {mem()}")


if __name__ == "__main__":
    main()
