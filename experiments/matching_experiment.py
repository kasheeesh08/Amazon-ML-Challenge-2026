#!/usr/bin/env python3
"""Step 3 — pairwise matcher: build, train, select and evaluate. No submission is produced.

Leakage-safe setup
  * Internal training set "A" is drawn from the FIT fold only: as many fit S1 as there are validation S1,
    with a pool = their true targets + the same number of fit distractors per source as the validation
    pool has. Candidates come from the unchanged Step 2 configuration, so training candidates are
    distributed like validation candidates.
  * A is split by S1 into A1 (80 %, model fitting) and A2 (20 %, threshold / model / rule selection).
  * VALIDATION (experiments/validation_split) is used only for the final evaluation: its candidates are
    read from experiments/blocking/val_candidate_pairs.tsv, its labels only in phase "evaluate".

Phases (each writes its outputs to --out-dir before the next starts; completed phases are reused):
  setup, candidates, idf, features, train, select, evaluate, errors, report       (or: all)

    python experiments/matching_experiment.py [--phase all]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.ids_tsv import GROUND_TRUTH_HEADER, MATCHING_HEADER, read_id_list_tsv  # noqa: E402
from ber.metrics import macro_f05  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402

SEED = 20260925
A2_FRACTION = 0.2
NEG_RATE = 0.10                    # training negatives kept (A1 only); weight 1/NEG_RATE preserves base rate
CHUNK_PAIRS = 2_000_000
GBDT_PARAMS = dict(n_trees=300, max_depth=6, learning_rate=0.1, lam=1.0, min_child_weight=20.0, seed=0)
GBDT_STAGES = [50, 100, 150, 200, 300]
THRESHOLDS = np.round(np.arange(0.02, 0.991, 0.01), 2)
PHASES = ["setup", "candidates", "idf", "features", "train", "select", "evaluate", "errors", "report"]


def log(*a):
    print(*a, flush=True)


def jdump(path, obj):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(conv(obj), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Run:
    def __init__(self, args):
        from blocking_experiment import Context  # Step 2 context: feature stores, truth, folds
        self.args = args
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(args.feat_dir, args.split_dir, args.cache_dir)
        self.config = jload(args.blocking_config)

    def p(self, name):
        return self.out / name

    def done(self, name):
        return self.p(name).exists()

    # --------------------------------------------------------------------------------------------
    def setup(self):
        ctx = self.ctx
        ids = {k: np.asarray(ctx.s[k].col("id")) for k in (1, 2, 3)}
        n_val = int((ctx.fold[1] == 1).sum())
        fit_s1 = np.flatnonzero(ctx.fold[1] == 0)
        h = seeded_hash([f"S1-{int(x)}" for x in ids[1][fit_s1]], SEED + 101)
        qA = np.sort(fit_s1[np.argsort(h, kind="stable")[:n_val]])
        # pool: targets of A + fit distractors (same count per source as the validation pool)
        posA = np.full(ctx.n1, -1, dtype=np.int64)
        posA[qA] = np.arange(len(qA))
        inA = posA[ctx.pair_s1row] >= 0
        pool, info = {}, {}
        for k in (2, 3):
            is_target = np.zeros(ctx.s[k].n, dtype=bool)
            is_target[ctx.pair_trow[ctx.pair_src == k]] = True
            n_val_distr = int(((ctx.fold[k] == 1) & ~is_target).sum())
            fit_distr = np.flatnonzero((ctx.fold[k] == 0) & ~is_target)
            hd = seeded_hash([f"S{k}-{int(x)}" for x in ids[k][fit_distr]], SEED + 102 + k)
            distr = fit_distr[np.argsort(hd, kind="stable")[:n_val_distr]]
            targets = np.unique(ctx.pair_trow[(ctx.pair_src == k) & inA])
            assert (ctx.fold[k][targets] == 0).all()
            pool[k] = np.sort(np.concatenate([targets, distr]))
            info[k] = {"targets": int(len(targets)), "distractors": int(len(distr)), "pool": int(len(pool[k]))}
        hA2 = seeded_hash([f"S1-{int(x)}" for x in ids[1][qA]], SEED + 103)
        a2 = np.zeros(len(qA), dtype=bool)
        a2[np.argsort(hA2, kind="stable")[:int(round(A2_FRACTION * len(qA)))]] = True
        np.save(self.p("qA_rows.npy"), qA)
        np.save(self.p("poolA_s2.npy"), pool[2])
        np.save(self.p("poolA_s3.npy"), pool[3])
        np.save(self.p("a2_mask.npy"), a2)
        assert (ctx.fold[1][qA] == 0).all() and not np.intersect1d(qA, np.flatnonzero(ctx.fold[1] == 1)).size
        jdump(self.p("setup.json"), {"n_A": int(len(qA)), "n_A1": int((~a2).sum()), "n_A2": int(a2.sum()),
                                     "pool": info, "val_s1": n_val, "seed": SEED})
        log(f"  A: {len(qA):,} fit S1 (A1 {(~a2).sum():,} / A2 {a2.sum():,}); pool {info}")

    def load_setup(self):
        qA = np.load(self.p("qA_rows.npy"))
        poolA = {2: np.load(self.p("poolA_s2.npy")), 3: np.load(self.p("poolA_s3.npy"))}
        return qA, poolA, np.load(self.p("a2_mask.npy"))

    def val_rows(self):
        ctx = self.ctx
        return np.flatnonzero(ctx.fold[1] == 1), {k: np.flatnonzero(ctx.fold[k] == 1) for k in (2, 3)}

    # --------------------------------------------------------------------------------------------
    def candidates(self):
        from blocking_experiment import codes_from_candidate_tsv, lean_union
        qA, poolA, _ = self.load_setup()
        trueA = self.ctx.truth(qA)
        if not self.done("cand_A.npy"):
            log("  blocking the internal training set A (Step 2 configuration, unchanged)")
            union, info = lean_union(self.ctx, self.config, qA, poolA, trueA)
            np.save(self.p("cand_A.npy"), union)
            jdump(self.p("cand_A_channels.json"), info)
        cA = np.load(self.p("cand_A.npy"))
        qv, _ = self.val_rows()
        if not self.done("cand_val.npy"):
            log("  reading validation candidates from the Step 2 candidate_pairs.tsv")
            np.save(self.p("cand_val.npy"), codes_from_candidate_tsv(self.ctx, self.args.val_candidates, qv))
        cv = np.load(self.p("cand_val.npy"))
        jdump(self.p("candidates.json"), {
            "A": {"pairs": int(len(cA)), "true_pairs": int(len(trueA)),
                  "candidate_recall": float(np.isin(trueA, cA, assume_unique=True).mean())},
            "val": {"pairs": int(len(cv))}})
        log(f"  A candidates {len(cA):,} (recall {np.isin(trueA, cA, assume_unique=True).mean():.4f}); "
            f"val candidates {len(cv):,}")

    # --------------------------------------------------------------------------------------------
    def idf(self):
        _, poolA = self.load_setup()[:2]
        _, poolV = self.val_rows()
        stores = {2: self.ctx.s[2], 3: self.ctx.s[3]}
        for tag, pool in (("A", poolA), ("val", poolV)):
            for ch in ("name", "addr"):
                MF.build_idf(stores, pool, ch).save(self.p(f"idf_{tag}_{ch}.npz"))
                log(f"  idf {tag}/{ch} saved")

    # --------------------------------------------------------------------------------------------
    def _features_for(self, tag, codes, q_rows):
        idf_n = MF.IdfTable.load(self.p(f"idf_{tag}_name.npz"))
        idf_a = MF.IdfTable.load(self.p(f"idf_{tag}_addr.npz"))
        n = len(codes)
        X = np.lib.format.open_memmap(self.p(f"X_{tag}.npy"), mode="w+", dtype=np.float32,
                                      shape=(n, MF.N_FEATURES))
        q = codes >> B.Q_SHIFT
        a = 0
        t0 = time.time()
        stores = {2: self.ctx.s[2], 3: self.ctx.s[3]}
        while a < n:
            b = min(n, a + CHUNK_PAIRS)
            if b < n:
                b = int(np.searchsorted(q, q[b], "left"))
                if b == a:
                    b = int(np.searchsorted(q, q[a], "right"))
            X[a:b] = MF.pair_features(self.ctx.s[1], q_rows, stores, codes[a:b], idf_n, idf_a)
            a = b
        cos = np.array(X[:, MF.COL["nameaddr_cos"]])
        r, g, s = MF.target_context(codes, cos)
        X[:, MF.COL["t_rank"]], X[:, MF.COL["t_gap"]], X[:, MF.COL["t_ncand"]] = r, g, s
        X.flush()
        del X
        log(f"  features {tag}: {n:,} pairs x {MF.N_FEATURES} ({time.time() - t0:.0f}s)")

    def features(self):
        qA, _, _ = self.load_setup()
        qv, _ = self.val_rows()
        for tag, q_rows in (("A", qA), ("val", qv)):
            codes = np.load(self.p(f"cand_{tag}.npy"))
            true = self.ctx.truth(q_rows)
            np.save(self.p(f"y_{tag}.npy"), np.isin(codes, true, assume_unique=True).astype(np.uint8))
            self._features_for(tag, codes, q_rows)

    # --------------------------------------------------------------------------------------------
    def train_rows(self):
        codes = np.load(self.p("cand_A.npy"))
        y = np.load(self.p("y_A.npy"))
        a2 = np.load(self.p("a2_mask.npy"))
        q = codes >> B.Q_SHIFT
        in_a1 = ~a2[q]
        with np.errstate(over="ignore"):
            hsh = ((codes.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
        keep_neg = hsh < np.uint64(int(NEG_RATE * 1000))
        rows = np.flatnonzero(in_a1 & ((y == 1) | keep_neg))
        w = np.where(y[rows] == 1, 1.0, 1.0 / NEG_RATE)
        return rows, y[rows], w, np.flatnonzero(a2[q])

    def train(self):
        rows, y, w, a2_rows = self.train_rows()
        XA = np.load(self.p("X_A.npy"), mmap_mode="r")
        yA = np.load(self.p("y_A.npy"))
        a2 = np.load(self.p("a2_mask.npy"))
        q = np.load(self.p("cand_A.npy")) >> B.Q_SHIFT
        dist = {"A_pairs": int(len(yA)), "A_pos": int(yA.sum()), "A_neg": int((yA == 0).sum()),
                "A1_pairs": int((~a2[q]).sum()), "A1_pos": int(yA[~a2[q]].sum()),
                "A2_pairs": int(a2[q].sum()), "A2_pos": int(yA[a2[q]].sum()),
                "train_rows": int(len(rows)), "train_pos": int(y.sum()), "train_neg": int((y == 0).sum()),
                "neg_rate": NEG_RATE, "neg_weight": 1.0 / NEG_RATE}
        log(f"  class distribution: {dist}")
        X = np.asarray(XA[rows])
        t0 = time.time()
        lr = M.LogReg(l2=1.0).fit(X, y, w)
        lr.save(self.p("model_logreg.npz"))
        log(f"  logistic regression fitted ({time.time() - t0:.0f}s, converged={lr.converged})")
        t0 = time.time()
        gb = GBDT(**GBDT_PARAMS).fit(X, y, w, log=log)
        gb.save(self.p("model_gbdt.npz"))
        log(f"  GBDT fitted ({time.time() - t0:.0f}s)")
        # A2 predictions (never used for fitting)
        XA2 = XA[a2_rows]
        np.save(self.p("pA2_logreg.npy"), lr.predict_proba(XA2))
        staged = gb.raw_staged(XA2, GBDT_STAGES)
        for s, raw in staged.items():
            np.save(self.p(f"pA2_gbdt_{s}.npy"), (1 / (1 + np.exp(-np.clip(raw, -35, 35)))).astype(np.float32))
        jdump(self.p("train.json"), {"class_distribution": dist, "gbdt_params": GBDT_PARAMS,
                                     "logreg_converged": lr.converged, "features": MF.FEATURE_NAMES})

    # --------------------------------------------------------------------------------------------
    def select(self):
        """Choose model, GBDT size, threshold and the one-to-one rule on A2 ONLY (official F0.5)."""
        qA, _, a2 = self.load_setup()
        codes = np.load(self.p("cand_A.npy"))
        q = codes >> B.Q_SHIFT
        a2_rows = np.flatnonzero(a2[q])
        c2 = codes[a2_rows]
        trueA = self.ctx.truth(qA)
        t2 = trueA[a2[trueA >> B.Q_SHIFT]]
        a2_idx = np.flatnonzero(a2)
        results = []
        variants = [("logreg", None)] + [("gbdt", s) for s in GBDT_STAGES]
        for model, stage in variants:
            p = np.load(self.p("pA2_logreg.npy" if model == "logreg" else f"pA2_gbdt_{stage}.npy"))
            for one in (False, True):
                curve = []
                for th in THRESHOLDS:
                    sel = c2[M.select(c2, p, th, one)]
                    _, parts = M.entity_f05(sel, t2, len(qA), return_parts=True)
                    curve.append(float(parts["F"][a2_idx].mean()))
                best = int(np.argmax(curve))
                results.append({"model": model, "n_trees": stage, "one_to_one": one,
                                "best_threshold": float(THRESHOLDS[best]), "best_f05": curve[best],
                                "curve": curve})
                log(f"    A2 {model}{'' if stage is None else '@' + str(stage)} 1:1={one}: "
                    f"best F0.5 {curve[best]:.5f} at t={THRESHOLDS[best]:.2f}")
        best = max(results, key=lambda r: (r["best_f05"], r["model"] == "logreg"))
        base = max((r for r in results if r["model"] == "logreg"), key=lambda r: r["best_f05"])
        jdump(self.p("selection.json"), {"thresholds": THRESHOLDS, "variants": results, "chosen": best,
                                         "baseline_logreg": base,
                                         "rule": "maximise official macro F0.5 on A2 (internal split of fit fold)"})
        log(f"  chosen on A2: {best['model']}{'' if best['n_trees'] is None else '@' + str(best['n_trees'])} "
            f"1:1={best['one_to_one']} t={best['best_threshold']} (A2 F0.5 {best['best_f05']:.5f})")

    # --------------------------------------------------------------------------------------------
    def _predict_val(self, sel_spec):
        XV = np.load(self.p("X_val.npy"), mmap_mode="r")
        if sel_spec["model"] == "logreg":
            return M.LogReg.load(self.p("model_logreg.npz")).predict_proba(XV)
        return GBDT.load(self.p("model_gbdt.npz")).predict_proba(XV, n_trees=sel_spec["n_trees"])

    def _metrics(self, codes, prob, spec, true, qv):
        nq = len(qv)
        sel_mask = M.select(codes, prob, spec["best_threshold"], spec["one_to_one"])
        pred = codes[sel_mask]
        blocked_true = true[np.isin(true, codes, assume_unique=True)]
        off, parts = M.entity_f05(pred, true, nq, return_parts=True)
        ntrue, npred = parts["ntrue"], parts["npred"]
        single = ntrue == 0
        res = {"official_macro_f05": off,
               "pair": M.pair_metrics(pred, true, blocked_true),
               "entity_precision_mean": float(parts["P"][npred > 0].mean()) if (npred > 0).any() else None,
               "entity_recall_mean": float(parts["R"][ntrue > 0].mean()),
               "f05_singletons": float(parts["F"][single].mean()),
               "f05_nonsingletons": float(parts["F"][~single].mean()),
               "n_singletons": int(single.sum()),
               "singleton_false_merge_rate": float((npred[single] > 0).mean()),
               "nonsingleton_empty_prediction_rate": float((npred[~single] == 0).mean()),
               "exact_set_match_rate": float(((parts["tp"] == ntrue) & (npred == ntrue)).mean()),
               "candidate_recall_ceiling": float(len(blocked_true) / len(true))}
        _, psrc, _ = B.decode(pred)
        _, tsrc, _ = B.decode(true)
        res["by_source"] = {}
        for s in (2, 3):
            ps, ts = pred[psrc == s], true[tsrc == s]
            f_s = M.entity_f05(ps, ts, nq)
            res["by_source"][f"S{s}"] = {"pair": M.pair_metrics(ps, ts, blocked_true[B.decode(blocked_true)[1] == s]),
                                         "macro_f05_restricted_to_source": f_s}
        ctry = np.asarray(self.ctx.s[1].col("country"))[qv]
        labels = {}
        for c in np.unique(ctry):
            m = ctry == c
            labels[int(c)] = {"n": int(m.sum()), "macro_f05": float(parts["F"][m].mean())}
        res["by_country_hash"] = labels
        return res, sel_mask

    def evaluate(self):
        qv, poolv = self.val_rows()
        codes = np.load(self.p("cand_val.npy"))
        true = self.ctx.truth(qv)
        S = jload(self.p("selection.json"))
        out = {}
        for name, spec in (("chosen", S["chosen"]), ("baseline_logreg", S["baseline_logreg"])):
            prob = self._predict_val(spec)
            np.save(self.p(f"pval_{name}.npy"), prob)
            res, sel_mask = self._metrics(codes, prob, spec, true, qv)
            res["spec"] = {k: spec[k] for k in ("model", "n_trees", "one_to_one", "best_threshold")}
            # threshold curve on validation — REPORTED ONLY, not used for any choice
            res["val_threshold_curve"] = [
                {"threshold": float(th), "macro_f05": M.entity_f05(codes[M.select(codes, prob, th, spec["one_to_one"])], true, len(qv))}
                for th in THRESHOLDS[::2]]
            out[name] = res
            log(f"  {name}: official macro F0.5 = {res['official_macro_f05']:.5f}; pair P={res['pair']['precision']:.4f} "
                f"R={res['pair']['recall']:.4f}; singleton F0.5={res['f05_singletons']:.4f}")
            if name == "chosen":
                np.save(self.p("val_selected_mask.npy"), sel_mask)
                self._write_and_check(codes[sel_mask], qv, out[name])
        # country hash -> label
        cmap = {}
        with open(Path(self.args.split_dir) / "val_as_test_dir" / "test_source1.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")[-1].strip()
                if c not in cmap.values():
                    from ber.normalize import h64
                    cmap[h64(c.casefold())] = c
                if len(cmap) >= 3:
                    break
        out["country_labels"] = {str(k): v for k, v in cmap.items()}
        jdump(self.p("val_eval.json"), out)

    def _write_and_check(self, pred_codes, qv, res):
        """Validation-only prediction file, checked by the official validator and the Step 1 scorer."""
        ids = {k: np.asarray(self.ctx.s[k].col("id")) for k in (1, 2, 3)}
        path = self.p("val_matching_predictions.tsv")
        M.write_id_lists(path, MATCHING_HEADER, pred_codes, [f"S1-{int(x)}" for x in ids[1][qv]],
                         {2: ids[2], 3: ids[3]})
        tdir = Path(self.args.split_dir) / "val_as_test_dir"
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        pr = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"), "--matching", str(path),
                             "--candidate", str(self.args.val_candidates), "--test-dir", str(tdir)],
                            capture_output=True, text=True, encoding="utf-8", env=env)
        res["official_validator"] = {"exit_code": pr.returncode, "tail": pr.stdout.strip().splitlines()[-3:]}
        from score import score_files
        rep = score_files(path, Path(self.args.split_dir) / "val_ground_truth.tsv", test_dir=str(tdir))
        res["step1_scorer_macro_f05"] = rep.macro_f05
        res["step1_scorer_matches_vectorised"] = abs(rep.macro_f05 - res["official_macro_f05"]) < 1e-12
        log(f"  official validator exit={pr.returncode}; Step 1 scorer macro F0.5 = {rep.macro_f05:.6f} "
            f"(matches vectorised: {res['step1_scorer_matches_vectorised']})")

    # --------------------------------------------------------------------------------------------
    def errors(self):
        from blocking_experiment import missed_examples  # noqa: F401  (raw-text fetch pattern)
        qv, _ = self.val_rows()
        codes = np.load(self.p("cand_val.npy"))
        y = np.load(self.p("y_val.npy")).astype(bool)
        sel = np.load(self.p("val_selected_mask.npy"))
        prob = np.load(self.p("pval_chosen.npy"))
        true = self.ctx.truth(qv)
        XV = np.load(self.p("X_val.npy"), mmap_mode="r")
        C = MF.COL
        fp = np.flatnonzero(sel & ~y)
        tp = np.flatnonzero(sel & y)
        fn_blocked = np.flatnonzero(~sel & y)
        not_blocked = true[~np.isin(true, codes, assume_unique=True)]
        ntrue = np.bincount(true >> B.Q_SHIFT, minlength=len(qv))
        q = codes >> B.Q_SHIFT

        def rates(idx):
            if len(idx) == 0:
                return {}
            Xs = np.asarray(XV[np.sort(idx)])
            f = lambda c: Xs[:, C[c]]  # noqa: E731
            return {
                "n": int(len(idx)),
                "same_normalised_name(eq_name_n2)": float(f("eq_name_n2").mean()),
                "same_normalised_address(eq_addr_n2)": float(f("eq_addr_n2").mean()),
                "name_cos>=0.8 & addr_cos<0.3 (same name, different place)":
                    float(((f("name_cos") >= 0.8) & (f("addr_cos") < 0.3)).mean()),
                "addr_cos>=0.8 & name_cos<0.3 (same place, different name)":
                    float(((f("addr_cos") >= 0.8) & (f("name_cos") < 0.3)).mean()),
                "house_both_present_but_differ": float(((f("house_both") == 1) & (f("eq_house") == 0)).mean()),
                "target_address_missing": float(f("t_addr_missing").mean()),
                "target_name_non_latin": float(f("t_name_nonlatin").mean()),
                "target_name_domain_like": float(f("t_name_domain").mean()),
                "target_name_junk_prefix": float(f("t_name_junk").mean()),
                "landmark_either_side": float(((f("t_landmark") + f("q_landmark")) > 0).mean()),
                "cross_country(eq_country==0)": float((f("eq_country") == 0).mean()),
                "query_is_singleton": float((ntrue[q[np.sort(idx)]] == 0).mean()),
                "mean_name_cos": float(f("name_cos").mean()), "mean_addr_cos": float(f("addr_cos").mean()),
                "mean_rank_src": float(f("rank_src").mean()), "mean_t_rank": float(f("t_rank").mean()),
            }
        E = {"counts": {"TP": int(len(tp)), "FP": int(len(fp)), "FN_blocked_rejected": int(len(fn_blocked)),
                        "FN_not_in_candidates": int(len(not_blocked)),
                        "FP_on_singleton_S1": int((ntrue[q[fp]] == 0).sum())},
             "profile": {"TP": rates(tp), "FP": rates(fp), "FN_blocked_rejected": rates(fn_blocked)}}
        gb = GBDT.load(self.p("model_gbdt.npz"))
        gi = gb.gain_importance / gb.gain_importance.sum()
        E["gbdt_gain_importance_top25"] = sorted(zip(MF.FEATURE_NAMES, gi.tolist()), key=lambda x: -x[1])[:25]
        lr = M.LogReg.load(self.p("model_logreg.npz"))
        E["logreg_std_coef_top15"] = sorted(zip(MF.FEATURE_NAMES, lr.coef.tolist()), key=lambda x: -abs(x[1]))[:15]
        # examples (actual validation records)
        rng = np.random.default_rng(SEED)
        pick = {
            "top_confidence_false_positives": fp[np.argsort(-prob[fp], kind="stable")[:12]],
            "singleton_false_merges": fp[ntrue[q[fp]] == 0][np.argsort(-prob[fp][ntrue[q[fp]] == 0], kind="stable")[:10]],
            "false_negatives_blocked_but_rejected": np.sort(rng.choice(fn_blocked, size=min(12, len(fn_blocked)), replace=False)) if len(fn_blocked) else fn_blocked,
        }
        E["examples"] = {k: self._examples(codes[v], prob[v], qv) for k, v in pick.items()}
        jdump(self.p("errors.json"), E)
        log(f"  errors: {E['counts']}")

    def _examples(self, codes, probs, qv):
        import eda
        q, src, t = B.decode(codes)
        s1r = qv[q]
        need = {1: set(s1r.tolist()), 2: set(t[src == 2].tolist()), 3: set(t[src == 3].tolist())}
        lines = {}
        for k in (1, 2, 3):
            got = {}
            if need[k]:
                last = max(need[k])
                with open(Path(self.args.data_dir) / f"train_source{k}.tsv", "rb") as f:
                    f.readline()
                    row = 0
                    for b in f:
                        if not b.strip():
                            continue
                        if row in need[k]:
                            got[row] = (b.decode("utf-8").rstrip("\r\n").split("\t") + [""] * 4)[:4]
                        if row >= last:
                            break
                        row += 1
            lines[k] = got
        out = []
        for a, s, b, pr in zip(s1r, src, t, probs):
            r1, r2 = lines[1][int(a)], lines[int(s)][int(b)]
            out.append({"prob": float(pr), "s1": r1, "candidate": r2, "differences": eda.pair_tags(r1, r2)[0]})
        return out

    # --------------------------------------------------------------------------------------------
    def report(self):
        write_report(self.out)


def fp_(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def write_report(out):
    out = Path(out)
    st, cand, tr = jload(out / "setup.json"), jload(out / "candidates.json"), jload(out / "train.json")
    S, V = jload(out / "selection.json"), jload(out / "val_eval.json")
    Er = jload(out / "errors.json") if (out / "errors.json").exists() else None
    L = []
    w = L.append
    ch = V["chosen"]
    spec = ch["spec"]
    w("# Step 3 — pairwise matcher report\n")
    w("Generated by `experiments/matching_experiment.py`. Validation labels are used only for the final evaluation; "
      "model, size, threshold and post-processing rule were chosen on A2 (internal split of the FIT fold). "
      "No submission file was produced.\n")
    w("## A. Model and features\n")
    w(f"- Candidates: Step 2 blocking (unchanged). Features: {len(tr['features'])} per pair, computed with numba from "
      "the Step 2 feature store (`src/ber/match_features.py`):")
    w("  " + ", ".join(f"`{f}`" for f in tr["features"]))
    w("- No phone / e-mail / website field exists in the data; domain-like names enter only as a flag.")
    w(f"- Models: weighted L2 logistic regression (baseline, `src/ber/matcher.py`) and a from-scratch numba histogram "
      f"GBDT (`src/ber/gbdt.py`; no GBDT library is installed) with {tr['gbdt_params']}.")
    w(f"- **Selected on A2: {spec['model']}{'' if spec['n_trees'] is None else ' with ' + str(spec['n_trees']) + ' trees'}, "
      f"threshold {spec['best_threshold']:.2f}, one-S1-per-target rule {'ON' if spec['one_to_one'] else 'OFF'}.**\n")
    d = tr["class_distribution"]
    w("## B. Training data (FIT fold only)\n")
    w(f"- Internal set A: {st['n_A']:,} fit S1 (A1 {st['n_A1']:,} for fitting, A2 {st['n_A2']:,} for selection); pool "
      f"S2 {st['pool']['2']['pool']:,} / S3 {st['pool']['3']['pool']:,} (targets + fit distractors, validation-sized).")
    w(f"- A candidates: {cand['A']['pairs']:,} pairs; candidate recall {fp_(cand['A']['candidate_recall'])} "
      f"(validation: {fp_(ch['candidate_recall_ceiling'])}).")
    w(f"- Class distribution in A: {d['A_pos']:,} positive / {d['A_neg']:,} negative "
      f"({100 * d['A_pos'] / d['A_pairs']:.2f}% positive, 1 : {d['A_neg'] / d['A_pos']:.1f}).")
    w(f"- Training rows (A1): all {d['train_pos']:,} positives + {d['train_neg']:,} negatives "
      f"(deterministic {100 * d['neg_rate']:.0f}% sample, weight {d['neg_weight']:.0f} so the weighted base rate equals A1's). "
      f"A2 ({d['A2_pairs']:,} pairs) is never used for fitting.\n")
    w(f"## C. Validation results (held-out; {st['val_s1']:,} S1)\n")
    w("| metric | chosen model | logistic-regression baseline |")
    w("|---|---:|---:|")
    b = V["baseline_logreg"]
    rows = [("**official macro F0.5 (Step 1 metric)**", "official_macro_f05"),
            ("pair precision", ("pair", "precision")), ("pair recall (end-to-end)", ("pair", "recall")),
            ("pair recall among blocked true pairs", ("pair", "recall_among_blocked")),
            ("pair F0.5", ("pair", "f05")), ("pair F1", ("pair", "f1")),
            ("entity precision (mean over S1 with a prediction)", "entity_precision_mean"),
            ("entity recall (mean over S1 with true matches)", "entity_recall_mean"),
            ("F0.5 on singleton S1", "f05_singletons"), ("singleton false-merge rate", "singleton_false_merge_rate"),
            ("F0.5 on S1 with matches", "f05_nonsingletons"),
            ("S1 with matches but empty prediction", "nonsingleton_empty_prediction_rate"),
            ("S1 predicted exactly right", "exact_set_match_rate")]

    def get(r, k):
        return r[k[0]][k[1]] if isinstance(k, tuple) else r[k]
    for lab, k in rows:
        a, c = get(ch, k), get(b, k)
        fmt = (lambda x: f"{x:.5f}") if k == "official_macro_f05" else fp_
        w(f"| {lab} | {fmt(a)} | {fmt(c)} |")
    w(f"\nPredicted pairs {ch['pair']['n_pred']:,} (TP {ch['pair']['tp']:,}) of {ch['pair']['n_true']:,} true pairs; "
      f"candidate-recall ceiling {fp_(ch['candidate_recall_ceiling'])}. Baseline spec: {b['spec']}.")
    w(f"\nCross-checks: official validator on the validation prediction file exit code "
      f"{ch['official_validator']['exit_code']} ({ch['official_validator']['tail'][-1] if ch['official_validator']['tail'] else ''}); "
      f"Step 1 scorer (`src/score.py`) macro F0.5 = {ch['step1_scorer_macro_f05']:.6f} — identical to the vectorised value: "
      f"{ch['step1_scorer_matches_vectorised']}.\n")
    cl = V.get("country_labels", {})
    w("By country: " + ", ".join(f"{cl.get(k, k)} {v['macro_f05']:.4f} (n={v['n']:,})" for k, v in ch["by_country_hash"].items()) + "\n")
    w("## D. Threshold analysis\n")
    w("Selection on A2 (internal split) — best official F0.5 per variant:\n")
    w("| model | trees | one-to-one | best threshold | A2 macro F0.5 |")
    w("|---|---:|---|---:|---:|")
    for r in S["variants"]:
        mark = " ← chosen" if (r["model"], r["n_trees"], r["one_to_one"]) == (S["chosen"]["model"], S["chosen"]["n_trees"], S["chosen"]["one_to_one"]) else ""
        w(f"| {r['model']} | {r['n_trees'] or '—'} | {r['one_to_one']} | {r['best_threshold']:.2f} | {r['best_f05']:.5f}{mark} |")
    th = S["thresholds"]
    cur = S["chosen"]["curve"]
    w("\nA2 threshold curve of the chosen variant (every 0.05):\n")
    w("| threshold | " + " | ".join(f"{th[i]:.2f}" for i in range(3, len(th), 5)) + " |")
    w("|---|" + "---:|" * len(range(3, len(th), 5)))
    w("| A2 F0.5 | " + " | ".join(f"{cur[i]:.4f}" for i in range(3, len(th), 5)) + " |")
    vc = ch["val_threshold_curve"]
    w("\nValidation curve of the chosen variant (reported only — NOT used to choose anything):\n")
    w("| threshold | " + " | ".join(f"{r['threshold']:.2f}" for r in vc[1::3]) + " |")
    w("|---|" + "---:|" * len(vc[1::3]))
    w("| val F0.5 | " + " | ".join(f"{r['macro_f05']:.4f}" for r in vc[1::3]) + " |")
    best_val = max(vc, key=lambda r: r["macro_f05"])
    w(f"\nBest threshold on validation would have been {best_val['threshold']:.2f} (F0.5 {best_val['macro_f05']:.5f}) vs "
      f"{ch['official_macro_f05']:.5f} at the A2-chosen {spec['best_threshold']:.2f} — the A2 choice transfers.\n")
    w("## E. Source 2 vs Source 3\n")
    w("| source | pair precision | pair recall | pair F0.5 | recall among blocked | macro F0.5 restricted to source (diagnostic) |")
    w("|---|---:|---:|---:|---:|---:|")
    for s, v in ch["by_source"].items():
        p = v["pair"]
        w(f"| {s} | {fp_(p['precision'])} | {fp_(p['recall'])} | {fp_(p['f05'])} | {fp_(p['recall_among_blocked'])} | "
          f"{v['macro_f05_restricted_to_source']:.4f} |")
    w("")
    if Er:
        c = Er["counts"]
        w("## F. Error analysis (validation, chosen model)\n")
        w(f"TP {c['TP']:,}; FP {c['FP']:,} (of which on singleton S1: {c['FP_on_singleton_S1']:,}); FN blocked-but-rejected "
          f"{c['FN_blocked_rejected']:,}; FN never in candidates (Step 2 ceiling) {c['FN_not_in_candidates']:,}.\n")
        keys = list(Er["profile"]["TP"].keys())
        w("| share of pairs with property | TP | FP | FN (blocked, rejected) |")
        w("|---|---:|---:|---:|")
        for k in keys:
            if k == "n":
                continue
            vals = [Er["profile"][g].get(k) for g in ("TP", "FP", "FN_blocked_rejected")]
            fmt = (lambda x: "—" if x is None else f"{x:.3f}") if k.startswith("mean") else fp_
            w(f"| {k} | " + " | ".join(fmt(x) for x in vals) + " |")
        w("\nGBDT gain importance (top 15): " + ", ".join(f"`{n}` {100 * g:.1f}%" for n, g in Er["gbdt_gain_importance_top25"][:15]))
        w("\nLogistic-regression standardised coefficients (top 10): " + ", ".join(f"`{n}` {v:+.2f}" for n, v in Er["logreg_std_coef_top15"][:10]) + "\n")
        w("## G. Examples (actual validation records)\n")
        for k, lst in Er["examples"].items():
            w(f"### {k.replace('_', ' ')}\n")
            for e in lst:
                s1, t = e["s1"], e["candidate"]
                w(f"- p={e['prob']:.3f}: **{s1[1]}** | {s1[2]} ({s1[3]})  ⟷  `{t[0]}` **{t[1]}** | {t[2] or '∅'} ({t[3]}) — "
                  f"{', '.join(e['differences'])}")
            w("")
    (out / "MATCHING_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {out / 'MATCHING_REPORT.md'}")


PHASE_OUTPUT = {"setup": "setup.json", "candidates": "candidates.json", "idf": "idf_val_addr.npz",
                "features": "X_val.npy", "train": "train.json", "select": "selection.json",
                "evaluate": "val_eval.json", "errors": "errors.json", "report": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "dataset" / "train"))
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--blocking-config", default=str(ROOT / "experiments" / "blocking" / "blocking_config.json"))
    ap.add_argument("--val-candidates", default=str(ROOT / "experiments" / "blocking" / "val_candidate_pairs.tsv"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "matching"))
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    ap.add_argument("--force", action="store_true", help="recompute the requested phase even if its output exists")
    args = ap.parse_args()
    t0 = time.time()
    if args.phase == "report":
        write_report(args.out_dir)
        return
    run = Run(args)
    for ph in PHASES:
        if args.phase not in ("all", ph):
            continue
        outp = PHASE_OUTPUT[ph]
        if outp and run.done(outp) and not (args.force and args.phase == ph):
            log(f"[{ph}] reuse ({outp} exists)")
            continue
        log(f"[{ph}]")
        tp = time.time()
        getattr(run, ph)()
        log(f"[{ph}] done ({time.time() - tp:.0f}s)")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
