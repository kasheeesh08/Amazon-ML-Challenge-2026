#!/usr/bin/env python3
"""READ-ONLY diagnostic: is the target-context feature ``t_rank`` behind the validation -> test gap?

Nothing in production is modified: Step 1-3 code/config/models, the final model, output/ and the submission package
are only read, and their SHA-256 are recorded before and after. All outputs go to experiments/diagnostics/trank/.

Environments (all use the validation S1 and validation labels only; no test data / labels):
  V  normal validation        validation S1 -> validation pool (saved Step 3 candidates + features)
  S  stress (existing)        validation S1 -> FULL training pool (saved stress candidates / IDF / target context)
  T  test-ratio (new)         validation S1 -> validation pool + random fit-fold records, sized so pool-per-S1 equals
                              the real test set per source (S1 per pool record 0.174 as in test). Blocking = unchanged
                              Step 2 config; IDF from this pool; features = unchanged Step 3 code.

Model variants (trained exactly like Step 3: A1 rows of the internal fit-fold set A, 300 trees, same params;
threshold chosen on A2 with the one-per-target rule, as in Step 3 — never on validation / stress labels):
  A   current feature set (the saved Step 3 model itself)
  B   t_rank removed
  B2  t_rank, t_gap, t_ncand removed (all target-context features)
  C   t_rank capped at 1 (binary "best S1 for this target"); monotone transforms are no-ops for trees
  D   t_rank replaced by its within-target percentile t_rank / max(t_ncand - 1, 1)

Phases: hashes_before, setup_T, candidates_T, idf_T, features_S, features_T, train, select, predict, evaluate,
breakdown, hashes_after, report
"""

import argparse
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
from ber.normalize import h64  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402
from matching_experiment import GBDT_PARAMS, NEG_RATE, THRESHOLDS  # noqa: E402
from test_scale_check import chunks_by_query, jdump, jload, sha256  # noqa: E402

SEED = 20260925
MD = ROOT / "experiments" / "matching"
SD = ROOT / "experiments" / "test_scale" / "stress"
TEST_POOL_PER_S1 = {2: 4_887_273 / 1_732_544, 3: 5_082_316 / 1_732_544}
N_TARGET_PARTS = 8
VARIANTS = ["A", "B", "B2", "C", "D"]
ENVS = ["V", "S", "T"]
C = MF.COL
KEEP_COLS = ["is_s3", "t_rank", "t_ncand", "name_cos", "addr_cos", "house_both", "eq_house", "t_addr_missing"]
ORACLE_GRID = np.round(np.arange(0.40, 0.991, 0.01), 2)

PROTECTED = (sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src" / "ber").glob("*.py"))
             + ["experiments/blocking/blocking_config.json", "experiments/matching/model_gbdt.npz",
                "experiments/matching/selection.json", "experiments/final/model_final.npz",
                "experiments/final_pipeline.py", "output/matching_results.tsv", "output/candidate_pairs.tsv",
                "camelcase_submission.zip"]
             + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "submission_package").rglob("*") if p.is_file()))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_ws_gb": round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2),
                "free_gb": round(psutil.virtual_memory().available / 1e9, 2)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------------------------------------------
# feature variants (applied to 50-column Step 3 feature blocks)
# ---------------------------------------------------------------------------------------------------------------
def transform(v, X):
    X = np.asarray(X, dtype=np.float32)
    if v == "A":
        return X
    if v == "B":
        return np.delete(X, [C["t_rank"]], axis=1)
    if v == "B2":
        return np.delete(X, [C["t_rank"], C["t_gap"], C["t_ncand"]], axis=1)
    Y = X.copy()
    if v == "C":
        Y[:, C["t_rank"]] = np.minimum(X[:, C["t_rank"]], 1.0)
    elif v == "D":
        Y[:, C["t_rank"]] = X[:, C["t_rank"]] / np.maximum(X[:, C["t_ncand"]] - 1.0, 1.0)
    return Y


def feature_names(v):
    names = list(MF.FEATURE_NAMES)
    if v == "B":
        names.remove("t_rank")
    elif v == "B2":
        for n in ("t_rank", "t_gap", "t_ncand"):
            names.remove(n)
    elif v == "C":
        names[C["t_rank"]] = "t_rank_cap1"
    elif v == "D":
        names[C["t_rank"]] = "t_rank_pct"
    return names


class Diag:
    def __init__(self, a):
        from blocking_experiment import Context
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(a.feat_dir, a.split_dir, a.cache_dir)
        self.qv = np.flatnonzero(self.ctx.fold[1] == 1)
        self.stores = {2: self.ctx.s[2], 3: self.ctx.s[3]}
        self.config = jload(a.blocking_config)
        self._true = None

    def p(self, n):
        return self.out / n

    @property
    def true(self):
        if self._true is None:
            self._true = self.ctx.truth(self.qv)
        return self._true

    def pool_T(self):
        return {k: np.load(self.p(f"poolT_s{k}.npy")) for k in (2, 3)}

    def env_paths(self, env):
        if env == "V":
            return MD / "cand_val.npy", MD / "X_val.npy"
        if env == "S":
            return SD / "cand_stress.npy", self.p("X_S.npy")
        return self.p("cand_T.npy"), self.p("X_T.npy")

    # -----------------------------------------------------------------------------------------------------------
    def hashes_before(self):
        jdump(self.p("hashes_before.json"), {f: sha256(ROOT / f) for f in PROTECTED})

    def hashes_after(self):
        hb = jload(self.p("hashes_before.json"))
        ha = {f: sha256(ROOT / f) for f in hb}
        jdump(self.p("hashes_after.json"), {"hashes": ha, "unchanged": all(ha[f] == hb[f] for f in hb),
                                            "n_files": len(hb)})
        log(f"  protected files unchanged: {all(ha[f] == hb[f] for f in hb)} ({len(hb)} files)")

    # -----------------------------------------------------------------------------------------------------------
    def setup_T(self):
        ctx, nq = self.ctx, len(self.qv)
        info = {}
        for k in (2, 3):
            val = np.flatnonzero(ctx.fold[k] == 1)
            n_add = max(0, int(round(TEST_POOL_PER_S1[k] * nq)) - len(val))
            fit = np.flatnonzero(ctx.fold[k] == 0)
            ids = np.asarray(ctx.s[k].col("id"))[fit]
            h = seeded_hash([f"S{k}-{int(x)}" for x in ids], SEED + 700 + k)
            add = np.sort(fit[np.argsort(h, kind="stable")[:n_add]])
            is_target = np.zeros(ctx.s[k].n, dtype=bool)
            is_target[ctx.pair_trow[ctx.pair_src == k]] = True
            np.save(self.p(f"poolT_s{k}.npy"), np.sort(np.concatenate([val, add])))
            info[f"S{k}"] = {"validation_pool": int(len(val)), "added_fit_records": int(n_add),
                             "added_that_are_fit_targets(orphans)": int(is_target[add].sum()),
                             "added_that_are_distractors": int((~is_target[add]).sum()),
                             "pool": int(len(val) + n_add)}
        tot = sum(v["pool"] for v in info.values())
        info["pool_per_s1"] = tot / nq
        info["s1_per_pool_record"] = nq / tot
        jdump(self.p("setup_T.json"), info)
        log(f"  T pool {info}")

    def candidates_T(self):
        from blocking_experiment import lean_union
        union, info = lean_union(self.ctx, self.config, self.qv, self.pool_T(), self.true)
        np.save(self.p("cand_T.npy"), union)
        rec = float(np.isin(self.true, union, assume_unique=True).mean())
        jdump(self.p("candidates_T.json"), {"pairs": int(len(union)), "candidate_recall": rec, "channels": info,
                                            "memory": mem()})
        log(f"  T candidates {len(union):,}; recall {rec:.5f}")

    def idf_T(self):
        for ch in ("name", "addr"):
            MF.build_idf(self.stores, self.pool_T(), ch).save(self.p(f"idf_T_{ch}.npz"))

    # -----------------------------------------------------------------------------------------------------------
    def _write_X(self, codes, idf_n, idf_a, path, tc=None):
        """Full 50-column feature matrix (memmap). ``tc``: precomputed target context arrays, else computed here."""
        n = len(codes)
        if tc is None:
            # pass 1: nameaddr_cos for every pair, then target context per target-row partition (as in the stress run)
            cos = np.empty(n, dtype=np.float32)
            for a, b in chunks_by_query(codes):
                cos[a:b] = MF.pair_features(self.ctx.s[1], self.qv, self.stores, codes[a:b], idf_n,
                                            idf_a)[:, C["nameaddr_cos"]]
            tc = {nm: np.zeros(n, dtype=np.float32) for nm in ("t_rank", "t_gap", "t_ncand")}
            _, s, t = B.decode(codes)
            P = N_TARGET_PARTS
            pid = (s - 2) * P + np.minimum((t * P) // np.where(s == 2, self.ctx.s[2].n, self.ctx.s[3].n), P - 1)
            order = np.argsort(pid, kind="stable")
            bnd = np.searchsorted(pid[order], np.arange(2 * P + 1))
            del s, t
            for k in range(2 * P):
                sel = np.sort(order[bnd[k]:bnd[k + 1]])
                if len(sel):
                    tc["t_rank"][sel], tc["t_gap"][sel], tc["t_ncand"][sel] = MF.target_context(codes[sel], cos[sel])
            del cos, order, pid
        X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, MF.N_FEATURES))
        for a, b in chunks_by_query(codes):
            blk = MF.pair_features(self.ctx.s[1], self.qv, self.stores, codes[a:b], idf_n, idf_a)
            for nm in ("t_rank", "t_gap", "t_ncand"):
                blk[:, C[nm]] = tc[nm][a:b]
            X[a:b] = blk
        X.flush()
        del X

    def features_S(self):
        codes = np.load(SD / "cand_stress.npy")
        idf_n, idf_a = (MF.IdfTable.load(SD / f"idf_full_{c}.npz") for c in ("name", "addr"))
        tc = {nm: np.load(SD / f"{nm}_stress.npy", mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        self._write_X(codes, idf_n, idf_a, self.p("X_S.npy"), tc)
        log(f"  X_S written ({len(codes):,} pairs) {mem()}")

    def features_T(self):
        codes = np.load(self.p("cand_T.npy"))
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_T_{c}.npz")) for c in ("name", "addr"))
        self._write_X(codes, idf_n, idf_a, self.p("X_T.npy"))
        log(f"  X_T written ({len(codes):,} pairs) {mem()}")

    # -----------------------------------------------------------------------------------------------------------
    def _a1_rows(self):
        codes = np.load(MD / "cand_A.npy")
        y = np.load(MD / "y_A.npy")
        a2 = np.load(MD / "a2_mask.npy")
        q = codes >> B.Q_SHIFT
        with np.errstate(over="ignore"):
            hsh = ((codes.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
        keep_neg = hsh < np.uint64(int(NEG_RATE * 1000))
        rows = np.flatnonzero(~a2[q] & ((y == 1) | keep_neg))
        return rows, y[rows], np.where(y[rows] == 1, 1.0, 1.0 / NEG_RATE), np.flatnonzero(a2[q])

    def model_path(self, v):
        return MD / "model_gbdt.npz" if v == "A" else self.p(f"model_{v}.npz")

    def train(self):
        rows, y, w, a2_rows = self._a1_rows()
        XA = np.load(MD / "X_A.npy", mmap_mode="r")
        info = jload(self.p("train.json")) if self.p("train.json").exists() else {}
        for v in VARIANTS:
            if v == "A" or self.p(f"pA2_{v}.npy").exists():
                continue
            t0 = time.time()
            X = transform(v, XA[rows])
            gb = GBDT(**GBDT_PARAMS).fit(X, y, w, log=log)
            del X
            gb.save(self.p(f"model_{v}.npz"))
            pa2 = np.empty(len(a2_rows), dtype=np.float32)
            for a in range(0, len(a2_rows), 2_000_000):
                pa2[a:a + 2_000_000] = gb.predict_proba(transform(v, XA[a2_rows[a:a + 2_000_000]]))
            np.save(self.p(f"pA2_{v}.npy"), pa2)
            gi = gb.gain_importance / gb.gain_importance.sum()
            info[v] = {"seconds": round(time.time() - t0, 1), "train_rows": int(len(rows)),
                       "n_features": int(len(gi)),
                       "gain_top8": sorted(zip(feature_names(v), gi.tolist()), key=lambda x: -x[1])[:8],
                       "memory": mem()}
            jdump(self.p("train.json"), info)
            log(f"  variant {v} trained ({info[v]['seconds']}s) top gain {info[v]['gain_top8'][:3]}")
        gbA = GBDT.load(MD / "model_gbdt.npz")
        gi = gbA.gain_importance / gbA.gain_importance.sum()
        info["A"] = {"saved_step3_model": True, "n_features": int(len(gi)),
                     "gain_top8": sorted(zip(feature_names("A"), gi.tolist()), key=lambda x: -x[1])[:8]}
        jdump(self.p("train.json"), info)

    def select(self):
        """Step 3 rule: 300 trees, one-per-target ON, threshold maximising official macro F0.5 on A2."""
        qA = np.load(MD / "qA_rows.npy")
        a2 = np.load(MD / "a2_mask.npy")
        codes = np.load(MD / "cand_A.npy")
        c2 = codes[a2[codes >> B.Q_SHIFT]]
        trueA = self.ctx.truth(qA)
        t2 = trueA[a2[trueA >> B.Q_SHIFT]]
        a2_idx = np.flatnonzero(a2)
        res = {}
        for v in VARIANTS:
            p = np.load(MD / "pA2_gbdt_300.npy") if v == "A" else np.load(self.p(f"pA2_{v}.npy"))
            curve = []
            for th in THRESHOLDS:
                sel = c2[M.select(c2, p, th, True)]
                _, parts = M.entity_f05(sel, t2, len(qA), return_parts=True)
                curve.append(float(parts["F"][a2_idx].mean()))
            b = int(np.argmax(curve))
            res[v] = {"threshold": float(THRESHOLDS[b]), "a2_f05": curve[b]}
            log(f"  {v}: A2 best F0.5 {curve[b]:.5f} at t={THRESHOLDS[b]:.2f}")
        ref = [r for r in jload(MD / "selection.json")["variants"]
               if r["model"] == "gbdt" and r["n_trees"] == 300 and r["one_to_one"]][0]
        res["A_matches_step3_selection"] = abs(res["A"]["threshold"] - ref["best_threshold"]) < 1e-9
        assert res["A_matches_step3_selection"], (res["A"], ref["best_threshold"])
        jdump(self.p("select.json"), res)

    # -----------------------------------------------------------------------------------------------------------
    def predict(self):
        models = {v: GBDT.load(self.model_path(v)) for v in VARIANTS}
        checks = {}
        for env in ENVS:
            cpath, xpath = self.env_paths(env)
            if all(self.p(f"p_{env}_{v}.npy").exists() for v in VARIANTS) and self.p(f"cols_{env}.npz").exists():
                continue
            codes = np.load(cpath)
            X = np.load(xpath, mmap_mode="r")
            n = len(codes)
            probs = {v: np.empty(n, dtype=np.float32) for v in VARIANTS}
            cols = {c: np.empty(n, dtype=np.float32) for c in KEEP_COLS}
            t0 = time.time()
            for a in range(0, n, 2_000_000):
                blk = np.asarray(X[a:a + 2_000_000], dtype=np.float32)
                for c in KEEP_COLS:
                    cols[c][a:a + len(blk)] = blk[:, C[c]]
                for v, m in models.items():
                    probs[v][a:a + len(blk)] = m.predict_proba(transform(v, blk))
            for v in VARIANTS:
                np.save(self.p(f"p_{env}_{v}.npy"), probs[v])
            np.savez(self.p(f"cols_{env}.npz"), **cols)
            ref = {"V": MD / "pval_chosen.npy", "S": SD / "prob_stress.npy"}.get(env)
            if ref is not None:
                d = float(np.abs(np.load(ref) - probs["A"]).max())
                checks[env] = {"max_abs_diff_vs_saved_production_probs": d, "reproduces": d < 1e-5}
                log(f"  {env}: model A reproduces saved probabilities: max |diff| = {d:.2e}")
            log(f"  predicted {env} ({n:,} pairs, {time.time() - t0:.0f}s) {mem()}")
        if checks:
            jdump(self.p("reproduction_checks.json"), checks)

    # -----------------------------------------------------------------------------------------------------------
    def _target_origin(self, codes):
        """0 = validation pool, 1 = fit-fold target of an un-queried S1 (orphan), 2 = fit-fold distractor."""
        _, s, t = B.decode(codes)
        org = np.zeros(len(codes), dtype=np.int8)
        for k in (2, 3):
            is_target = np.zeros(self.ctx.s[k].n, dtype=bool)
            is_target[self.ctx.pair_trow[self.ctx.pair_src == k]] = True
            m = s == k
            fit = self.ctx.fold[k][t[m]] == 0
            o = np.zeros(m.sum(), dtype=np.int8)
            o[fit & is_target[t[m]]] = 1
            o[fit & ~is_target[t[m]]] = 2
            org[m] = o
        return org

    def _metrics(self, codes, prob, th, y, ctry, org):
        true, nq = self.true, len(self.qv)
        keep = M.select(codes, prob, th, True)
        pred = codes[keep]
        off, parts = M.entity_f05(pred, true, nq, return_parts=True)
        pm = M.pair_metrics(pred, true)
        single = parts["ntrue"] == 0
        tp = pm["tp"]
        fpm = keep & ~y
        res = {"threshold": th, "macro_f05": off, "precision": pm["precision"], "recall": pm["recall"],
               "singleton_f05": float(parts["F"][single].mean()),
               "nonsingleton_f05": float(parts["F"][~single].mean()),
               "singleton_false_merge_rate": float((parts["npred"][single] > 0).mean()),
               "pred": pm["n_pred"], "tp": tp, "fp": pm["n_pred"] - tp, "fn": pm["n_true"] - tp,
               "fn_not_in_candidates": int(pm["n_true"] - y.sum()),
               "fp_by_target_origin": {"validation_pool": int((fpm & (org == 0)).sum()),
                                       "fit_target_orphan": int((fpm & (org == 1)).sum()),
                                       "fit_distractor": int((fpm & (org == 2)).sum())},
               "by_country": {c: float(parts["F"][ctry == h].mean()) for c, h in self.countries().items()}}
        return res, keep, parts

    def countries(self):
        return {c: h64(c.casefold()) for c in ("US", "India")}

    def evaluate(self):
        sel = jload(self.p("select.json"))
        ctry = np.asarray(self.ctx.s[1].col("country"))[self.qv]
        out = {}
        for env in ENVS:
            codes = np.load(self.env_paths(env)[0])
            y = np.isin(codes, self.true, assume_unique=True)
            org = self._target_origin(codes)
            out[env] = {"pairs": int(len(codes)), "candidate_recall": float(y.sum() / len(self.true)),
                        "variants": {}}
            for v in VARIANTS:
                prob = np.load(self.p(f"p_{env}_{v}.npy"))
                res, _, _ = self._metrics(codes, prob, sel[v]["threshold"], y, ctry, org)
                # diagnostic only: best threshold on THIS environment's labels (not a selection)
                pre = np.flatnonzero(prob >= ORACLE_GRID[0])
                cp, pp = codes[pre], prob[pre]
                best = (-1.0, None)
                for th in ORACLE_GRID:
                    f = M.entity_f05(cp[M.select(cp, pp, th, True)], self.true, len(self.qv))
                    if f > best[0]:
                        best = (f, float(th))
                res["oracle_threshold"], res["oracle_macro_f05"] = best[1], best[0]
                out[env]["variants"][v] = res
                log(f"  {env}/{v}: F0.5 {res['macro_f05']:.5f} P {res['precision']:.4f} R {res['recall']:.4f} "
                    f"single {res['singleton_f05']:.4f} FP {res['fp']:,} FN {res['fn']:,} | oracle t={best[1]} "
                    f"F0.5 {best[0]:.5f}")
            jdump(self.p("evaluate.json"), out)

    # -----------------------------------------------------------------------------------------------------------
    def breakdown(self):
        sel = jload(self.p("select.json"))
        ctry = np.asarray(self.ctx.s[1].col("country"))[self.qv]
        ntrue = np.bincount(self.true >> B.Q_SHIFT, minlength=len(self.qv))
        out = {}
        for env in ENVS:
            codes = np.load(self.env_paths(env)[0])
            y = np.isin(codes, self.true, assume_unique=True)
            org = self._target_origin(codes)
            cols = np.load(self.p(f"cols_{env}.npz"))
            q = codes >> B.Q_SHIFT
            tr = cols["t_rank"]
            cats = {
                "t_rank=0": tr == 0, "t_rank=1": tr == 1, "t_rank=2": tr == 2, "t_rank=3-4": (tr >= 3) & (tr <= 4),
                "t_rank=5-9": (tr >= 5) & (tr <= 9), "t_rank>=10": tr >= 10,
                "t_ncand=1 (uncontested target)": cols["t_ncand"] == 1,
                "source S2": cols["is_s3"] == 0, "source S3": cols["is_s3"] == 1,
                "country US": ctry[q] == h64("us"), "country India": ctry[q] == h64("india"),
                "query singleton": ntrue[q] == 0, "query non-singleton": ntrue[q] > 0,
                "target address empty": cols["t_addr_missing"] == 1,
                "target address non-empty": cols["t_addr_missing"] == 0,
                "same name / different address (name_cos>=0.8 & addr_cos<0.3)":
                    (cols["name_cos"] >= 0.8) & (cols["addr_cos"] < 0.3),
                "different house number (both present, differ)": (cols["house_both"] == 1) & (cols["eq_house"] == 0),
                "target origin: validation pool": org == 0,
                "target origin: fit target (orphan)": org == 1,
                "target origin: fit distractor": org == 2,
            }
            out[env] = {}
            for v in VARIANTS:
                keep = M.select(codes, np.load(self.p(f"p_{env}_{v}.npy")), sel[v]["threshold"], True)
                rows = {}
                for name, m in cats.items():
                    n_c = int(m.sum())
                    if n_c == 0:
                        continue
                    tp = int((m & keep & y).sum())
                    fp = int((m & keep & ~y).sum())
                    pos = int((m & y).sum())
                    rows[name] = {"candidates": n_c, "true_in_candidates": pos, "tp": tp, "fp": fp,
                                  "fn_blocked": pos - tp, "precision": tp / (tp + fp) if tp + fp else None,
                                  "recall_blocked": tp / pos if pos else None}
                _, parts = M.entity_f05(codes[keep], self.true, len(self.qv), return_parts=True)
                ent = {"singleton": float(parts["F"][ntrue == 0].mean()),
                       "non-singleton": float(parts["F"][ntrue > 0].mean())}
                for c, h in self.countries().items():
                    ent[f"country {c}"] = float(parts["F"][ctry == h].mean())
                out[env][v] = {"pair_categories": rows, "entity_macro_f05": ent}
            jdump(self.p("breakdown.json"), out)
            log(f"  breakdown {env} done")

    # -----------------------------------------------------------------------------------------------------------
    def report(self):
        write_report(self)


def f5(x):
    return "—" if x is None else f"{x:.4f}"


def pc(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def write_report(d):
    E, S, Bk = jload(d.p("evaluate.json")), jload(d.p("select.json")), jload(d.p("breakdown.json"))
    Tr, ST, CT = jload(d.p("train.json")), jload(d.p("setup_T.json")), jload(d.p("candidates_T.json"))
    H = jload(d.p("hashes_after.json")) if d.p("hashes_after.json").exists() else {}
    R = jload(d.p("reproduction_checks.json")) if d.p("reproduction_checks.json").exists() else {}
    L = []
    w = L.append
    w("# t_rank / target-context diagnostic (read-only)\n")
    w("Generated by `experiments/trank_diagnostic.py`. Validation labels only; no test labels exist. Nothing in production "
      "was modified; no model was selected; no submission was produced.\n")
    w("## Environments\n")
    w("| env | description | S1 | candidate pairs | candidate recall |")
    w("|---|---|---:|---:|---:|")
    desc = {"V": "normal validation (validation pool)", "S": "stress: full training pool (S1/pool 0.043)",
            "T": f"test-ratio: validation pool + random fit records (S1/pool {ST['s1_per_pool_record']:.4f}; test 0.1738)"}
    for env in ENVS:
        w(f"| {env} | {desc[env]} | {len(d.qv):,} | {E[env]['pairs']:,} | {pc(E[env]['candidate_recall'])} |")
    w(f"\nT pool: {ST}\n")
    if R:
        w(f"Reproduction check (model A vs saved production probabilities): {R}\n")
    w("## Variants\n")
    w("A current; B −t_rank; B2 −t_rank −t_gap −t_ncand; C t_rank capped at 1; D t_rank → t_rank/max(t_ncand−1,1). "
      "All trained like Step 3 (A1 rows, 300 trees); threshold chosen on A2 with the one-per-target rule.\n")
    w("| variant | A2 threshold | A2 F0.5 | top gain features |")
    w("|---|---:|---:|---|")
    for v in VARIANTS:
        w(f"| {v} | {S[v]['threshold']:.2f} | {S[v]['a2_f05']:.5f} | "
          + ", ".join(f"`{n}` {100 * g:.1f}%" for n, g in Tr[v]["gain_top8"][:4]) + " |")
    w("\n## Main results (A2-chosen threshold)\n")
    w("| env | variant | macro F0.5 | precision | recall | singleton F0.5 | FP | FN | FP on orphans | oracle t | oracle F0.5 |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for env in ENVS:
        for v in VARIANTS:
            r = E[env]["variants"][v]
            w(f"| {env} | {v} | **{r['macro_f05']:.5f}** | {pc(r['precision'])} | {pc(r['recall'])} | "
              f"{pc(r['singleton_f05'])} | {r['fp']:,} | {r['fn']:,} | {r['fp_by_target_origin']['fit_target_orphan']:,} | "
              f"{r['oracle_threshold']} | {r['oracle_macro_f05']:.5f} |")
    w("\nFN includes pairs never blocked (V/S/T: " + ", ".join(
        f"{env} {E[env]['variants']['A']['fn_not_in_candidates']:,}" for env in ENVS) + ").\n")
    w("By country (macro F0.5): " + "; ".join(
        f"{env}/{v} " + ", ".join(f"{c} {x:.4f}" for c, x in E[env]["variants"][v]["by_country"].items())
        for env in ENVS for v in ("A", "B2")) + "\n")
    for env in ("S", "T"):
        w(f"## Breakdown — env {env}, current model A (pair level)\n")
        rows = Bk[env]["A"]["pair_categories"]
        w("| category | candidates | true in cand. | TP | FP | FN (blocked) | precision | recall (blocked) | FP with B2 | FN with B2 |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rb = Bk[env]["B2"]["pair_categories"]
        for k, r in rows.items():
            w(f"| {k} | {r['candidates']:,} | {r['true_in_candidates']:,} | {r['tp']:,} | {r['fp']:,} | {r['fn_blocked']:,} | "
              f"{pc(r['precision'])} | {pc(r['recall_blocked'])} | {rb[k]['fp']:,} | {rb[k]['fn_blocked']:,} |")
        w("\nEntity macro F0.5: " + "; ".join(
            f"{v}: " + ", ".join(f"{k} {x:.4f}" for k, x in Bk[env][v]["entity_macro_f05"].items()) for v in VARIANTS) + "\n")
    w(f"## Integrity\n\nProtected files unchanged: **{H.get('unchanged')}** ({H.get('n_files')} files: src/ber, blocking "
      "config, Step 3 model/selection, final model, output/, submission_package/, camelcase_submission.zip).\n")
    w(f"T candidates channels/recall: {CT['pairs']:,} pairs, recall {CT['candidate_recall']:.5f}.\n")
    (d.out / "TRANK_DIAGNOSTIC_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {d.out / 'TRANK_DIAGNOSTIC_REPORT.md'}")


PHASES = ["hashes_before", "setup_T", "candidates_T", "idf_T", "features_S", "features_T", "train", "select",
          "predict", "evaluate", "breakdown", "hashes_after", "report"]
PHASE_OUT = {"hashes_before": "hashes_before.json", "setup_T": "setup_T.json", "candidates_T": "candidates_T.json",
             "idf_T": "idf_T_addr.npz", "features_S": "X_S.done", "features_T": "X_T.done", "train": "train.done",
             "select": "select.json", "predict": "predict.done", "evaluate": "evaluate.json",
             "breakdown": "breakdown.json", "hashes_after": None, "report": None}


def main():
    global MD, SD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--blocking-config", default=str(ROOT / "experiments" / "blocking" / "blocking_config.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "diagnostics" / "trank"))
    ap.add_argument("--matching-dir", default=str(MD))
    ap.add_argument("--stress-dir", default=str(SD))
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    a = ap.parse_args()
    MD, SD = Path(a.matching_dir), Path(a.stress_dir)
    d = Diag(a)
    tp = d.p("timings.json")
    timings = jload(tp) if tp.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and d.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(d, ph)()
        if o and o.endswith(".done"):
            d.p(o).write_text("ok", encoding="utf-8")
        timings[ph] = round(time.time() - t0, 1)
        jdump(tp, timings)
        log(f"[{ph}] done ({timings[ph]}s) {mem()}")


if __name__ == "__main__":
    main()
