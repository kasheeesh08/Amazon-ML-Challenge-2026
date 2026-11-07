#!/usr/bin/env python3
"""Orphan-aware matcher training (DIAGNOSTIC; production untouched).

Baseline (A): the saved Step 3 model, trained on the internal fit-fold set A whose pool = A's own true targets + fit
distractors (every target's owner is queried -> no orphans are ever seen in training).

Orphan-aware (B): identical protocol, but A's pool is extended with fit-fold TARGET records whose owner S1 is NOT in
A (orphans, the same notion as in the stress diagnostic), sized per source so that pool records per S1 equal the real
test ratio (the same sizing as the test-ratio environment T). Candidates are regenerated with the unchanged Step 2
configuration, IDF is taken from the extended pool (Step 3 design), features/target context with the unchanged Step 3
code; training rows = A1 S1, all positives + the same deterministic 10 % hashed negatives (weight 10); GBDT with the
exact Step 3 parameters; threshold chosen on A2 (one-per-target rule on) as in Step 3, and also reported at 0.69.
Only training-fold labels are used. Evaluation on the SAVED labelled environments V / T / S (features unchanged).

Phases: hashes_before, setup, candidates, idf, features, train, select, predict, evaluate, hashes_after
"""

import argparse
import ctypes
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
from ber.gbdt import GBDT  # noqa: E402
from ber.validation_split import seeded_hash  # noqa: E402
from matching_experiment import GBDT_PARAMS, NEG_RATE, THRESHOLDS  # noqa: E402
from test_scale_check import chunks_by_query, jdump, jload, sha256  # noqa: E402
from trank_diagnostic import PROTECTED  # noqa: E402

SEED = 20260925
TEST_POOL_PER_S1 = {2: 4_887_273 / 1_732_544, 3: 5_082_316 / 1_732_544}
N_TARGET_PARTS = 8
C = MF.COL
ROOT_EXP = ROOT / "experiments"
MD, TD, SD, FD = (ROOT_EXP / "matching", ROOT_EXP / "diagnostics" / "trank", ROOT_EXP / "test_scale" / "stress",
                  ROOT_EXP / "final")
PBINS = [0.0, 0.5, 0.69, 0.8, 0.9, 0.95, 0.99, 1.0001]


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


def hash_tree(d):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(Path(d).rglob("*")) if p.is_file()}


class Exp:
    def __init__(self, a):
        from blocking_experiment import Context
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ctx = Context(a.feat_dir, a.split_dir, a.cache_dir)
        self.config = jload(a.blocking_config)
        self.qA = np.load(MD / "qA_rows.npy")
        self.a2 = np.load(MD / "a2_mask.npy")
        self.stores = {2: self.ctx.s[2], 3: self.ctx.s[3]}

    def p(self, n):
        return self.out / n

    def pool(self):
        return {k: np.load(self.p(f"poolB_s{k}.npy")) for k in (2, 3)}

    # --------------------------------------------------------------------------------------------------------------
    def hashes_before(self):
        jdump(self.p("hashes_before.json"), {"protected": {f: sha256(ROOT / f) for f in PROTECTED},
                                             "experiments_final": hash_tree(FD)})

    def hashes_after(self):
        hb = jload(self.p("hashes_before.json"))
        ha = {f: sha256(ROOT / f) for f in hb["protected"]}
        hf = hash_tree(FD)
        res = {"protected_unchanged": ha == hb["protected"], "n_protected": len(ha),
               "experiments_final_unchanged": hf == hb["experiments_final"], "n_final_files": len(hf),
               "output/matching_results.tsv": ha.get("output/matching_results.tsv"),
               "output/candidate_pairs.tsv": ha.get("output/candidate_pairs.tsv"),
               "experiments/final/model_final.npz": hf.get("experiments/final/model_final.npz")}
        jdump(self.p("hashes_after.json"), res)
        log(f"  integrity {res}")

    # --------------------------------------------------------------------------------------------------------------
    def setup(self):
        ctx, qA = self.ctx, self.qA
        nA = len(qA)
        inA = np.zeros(ctx.n1, dtype=bool)
        inA[qA] = True
        info = {}
        for k in (2, 3):
            baseA = np.load(MD / f"poolA_s{k}.npy")
            m = ctx.pair_src == k
            trow, owner = ctx.pair_trow[m], ctx.pair_s1row[m]
            orphan_rows = np.unique(trow[~inA[owner] & (ctx.fold[k][trow] == 0)])
            orphan_rows = orphan_rows[~np.isin(orphan_rows, baseA)]
            n_add = max(0, int(round(TEST_POOL_PER_S1[k] * nA)) - len(baseA))
            ids = np.asarray(ctx.s[k].col("id"))[orphan_rows]
            h = seeded_hash([f"S{k}-{int(x)}" for x in ids], SEED + 800 + k)
            add = np.sort(orphan_rows[np.argsort(h, kind="stable")[:n_add]])
            assert (ctx.fold[k][add] == 0).all()
            np.save(self.p(f"poolB_s{k}.npy"), np.sort(np.concatenate([baseA, add])))
            info[f"S{k}"] = {"base_pool_A": int(len(baseA)), "orphans_available": int(len(orphan_rows)),
                             "orphans_added": int(len(add)), "pool_B": int(len(baseA) + len(add))}
        tot = sum(v["pool_B"] for v in info.values())
        info["pool_per_s1"] = tot / nA
        info["s1_per_pool_record"] = nA / tot
        info["validation_records_used"] = 0
        jdump(self.p("setup.json"), info)
        log(f"  setup {info}")

    def candidates(self):
        from blocking_experiment import lean_union
        trueA = self.ctx.truth(self.qA)
        union, info = lean_union(self.ctx, self.config, self.qA, self.pool(), trueA)
        np.save(self.p("cand_B.npy"), union)
        base = np.load(MD / "cand_A.npy")
        jdump(self.p("candidates.json"), {"pairs_B": int(len(union)), "pairs_A_baseline": int(len(base)),
                                          "candidate_recall_B": float(np.isin(trueA, union, assume_unique=True).mean()),
                                          "candidate_recall_A": float(np.isin(trueA, base, assume_unique=True).mean()),
                                          "channels": info, "memory": mem()})
        log(f"  candidates B {len(union):,} (A {len(base):,})")

    def idf(self):
        for ch in ("name", "addr"):
            MF.build_idf(self.stores, self.pool(), ch).save(self.p(f"idf_B_{ch}.npz"))

    def features(self):
        codes = np.load(self.p("cand_B.npy"))
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_B_{c}.npz")) for c in ("name", "addr"))
        n = len(codes)
        cos = np.empty(n, dtype=np.float32)
        for a, b in chunks_by_query(codes):
            cos[a:b] = MF.pair_features(self.ctx.s[1], self.qA, self.stores, codes[a:b], idf_n, idf_a)[:, C["nameaddr_cos"]]
        tc = {nm: np.zeros(n, dtype=np.float32) for nm in ("t_rank", "t_gap", "t_ncand")}
        _, s, t = B.decode(codes)
        P = N_TARGET_PARTS
        pid = (s - 2) * P + np.minimum((t * P) // np.where(s == 2, self.ctx.s[2].n, self.ctx.s[3].n), P - 1)
        order = np.argsort(pid, kind="stable")
        bnd = np.searchsorted(pid[order], np.arange(2 * P + 1))
        for k in range(2 * P):
            sel = np.sort(order[bnd[k]:bnd[k + 1]])
            if len(sel):
                tc["t_rank"][sel], tc["t_gap"][sel], tc["t_ncand"][sel] = MF.target_context(codes[sel], cos[sel])
        del cos, order, pid, s, t
        X = np.lib.format.open_memmap(self.p("X_B.npy"), mode="w+", dtype=np.float32, shape=(n, MF.N_FEATURES))
        for a, b in chunks_by_query(codes):
            blk = MF.pair_features(self.ctx.s[1], self.qA, self.stores, codes[a:b], idf_n, idf_a)
            for nm in tc:
                blk[:, C[nm]] = tc[nm][a:b]
            X[a:b] = blk
        X.flush()
        del X
        trueA = self.ctx.truth(self.qA)
        np.save(self.p("y_B.npy"), np.isin(codes, trueA, assume_unique=True).astype(np.uint8))
        log(f"  X_B written ({n:,} pairs) {mem()}")

    def train(self):
        codes = np.load(self.p("cand_B.npy"))
        y = np.load(self.p("y_B.npy"))
        q = codes >> B.Q_SHIFT
        with np.errstate(over="ignore"):
            hsh = ((codes.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
        keep_neg = hsh < np.uint64(int(NEG_RATE * 1000))
        rows = np.flatnonzero(~self.a2[q] & ((y == 1) | keep_neg))
        w = np.where(y[rows] == 1, 1.0, 1.0 / NEG_RATE)
        XB = np.load(self.p("X_B.npy"), mmap_mode="r")
        t0 = time.time()
        gb = GBDT(**GBDT_PARAMS).fit(np.asarray(XB[rows]), y[rows], w, log=log)
        gb.save(self.p("model_orphan_aware.npz"))
        a2_rows = np.flatnonzero(self.a2[q])
        pa2 = np.empty(len(a2_rows), dtype=np.float32)
        for a in range(0, len(a2_rows), 2_000_000):
            pa2[a:a + 2_000_000] = gb.predict_proba(np.asarray(XB[a2_rows[a:a + 2_000_000]]))
        np.save(self.p("pA2_B.npy"), pa2)
        gi = gb.gain_importance / gb.gain_importance.sum()
        gA = GBDT.load(MD / "model_gbdt.npz")
        giA = gA.gain_importance / gA.gain_importance.sum()
        orph = np.zeros(len(codes), dtype=bool)
        _, s, t = B.decode(codes)
        for k in (2, 3):
            base = np.load(MD / f"poolA_s{k}.npy")
            mk = s == k
            orph[mk] = ~np.isin(t[mk], base)
        jdump(self.p("train.json"), {
            "seconds": round(time.time() - t0, 1), "train_rows": int(len(rows)), "train_pos": int(y[rows].sum()),
            "train_neg": int((y[rows] == 0).sum()), "train_rows_on_orphan_targets": int(orph[rows].sum()),
            "train_rows_on_orphan_targets_with_t_rank0": int((orph[rows] & (np.asarray(XB[rows][:, C["t_rank"]]) == 0)).sum()),
            "gbdt_params": GBDT_PARAMS,
            "gain_top10_B": sorted(zip(MF.FEATURE_NAMES, gi.tolist()), key=lambda x: -x[1])[:10],
            "gain_top10_A": sorted(zip(MF.FEATURE_NAMES, giA.tolist()), key=lambda x: -x[1])[:10],
            "target_context_gain_B": float(gi[[C["t_rank"], C["t_gap"], C["t_ncand"]]].sum()),
            "target_context_gain_A": float(giA[[C["t_rank"], C["t_gap"], C["t_ncand"]]].sum()), "memory": mem()})
        log(f"  trained B ({time.time() - t0:.0f}s, {len(rows):,} rows)")

    def select(self):
        codes = np.load(self.p("cand_B.npy"))
        q = codes >> B.Q_SHIFT
        c2 = codes[self.a2[q]]
        trueA = self.ctx.truth(self.qA)
        t2 = trueA[self.a2[trueA >> B.Q_SHIFT]]
        idx = np.flatnonzero(self.a2)
        p = np.load(self.p("pA2_B.npy"))
        curve = []
        for th in THRESHOLDS:
            _, parts = M.entity_f05(c2[M.select(c2, p, th, True)], t2, len(self.qA), return_parts=True)
            curve.append(float(parts["F"][idx].mean()))
        b = int(np.argmax(curve))
        jdump(self.p("select.json"), {"threshold_B": float(THRESHOLDS[b]), "a2B_f05": curve[b],
                                      "threshold_A": 0.69, "curve": curve, "thresholds": THRESHOLDS})
        log(f"  B: A2' best F0.5 {curve[b]:.5f} at t={THRESHOLDS[b]:.2f}")

    # --------------------------------------------------------------------------------------------------------------
    def envs(self):
        return {"V": (MD / "cand_val.npy", MD / "X_val.npy"), "T": (TD / "cand_T.npy", TD / "X_T.npy"),
                "S": (SD / "cand_stress.npy", TD / "X_S.npy")}

    def predict(self):
        gb = GBDT.load(self.p("model_orphan_aware.npz"))
        for env, (_, xp) in self.envs().items():
            if self.p(f"p_{env}_B.npy").exists():
                continue
            X = np.load(xp, mmap_mode="r")
            pr = np.empty(X.shape[0], dtype=np.float32)
            for a in range(0, X.shape[0], 2_000_000):
                pr[a:a + 2_000_000] = gb.predict_proba(np.asarray(X[a:a + 2_000_000]))
            np.save(self.p(f"p_{env}_B.npy"), pr)
            log(f"  predicted {env} ({X.shape[0]:,}) {mem()}")

    def evaluate(self):
        ctx = self.ctx
        qv = np.flatnonzero(ctx.fold[1] == 1)
        nq = len(qv)
        true = ctx.truth(qv)
        ntrue = np.bincount(true >> B.Q_SHIFT, minlength=nq)
        sel = jload(self.p("select.json"))
        is_t = {k: np.zeros(ctx.s[k].n, dtype=bool) for k in (2, 3)}
        for k in (2, 3):
            is_t[k][ctx.pair_trow[ctx.pair_src == k]] = True
        out = {}
        for env, (cp, xp) in self.envs().items():
            codes = np.load(cp)
            q, s, t = B.decode(codes)
            y = np.isin(codes, true, assume_unique=True)
            fit = np.zeros(len(codes), dtype=bool)
            tgt = np.zeros(len(codes), dtype=bool)
            for k in (2, 3):
                mk = s == k
                fit[mk] = ctx.fold[k][t[mk]] == 0
                tgt[mk] = is_t[k][t[mk]]
            X = np.load(xp, mmap_mode="r")
            tf = {nm: np.empty(len(codes), dtype=np.float32) for nm in ("t_rank", "t_gap", "t_ncand")}
            for a in range(0, len(codes), 2_000_000):
                blk = np.asarray(X[a:a + 2_000_000][:, [C["t_rank"], C["t_gap"], C["t_ncand"]]])
                tf["t_rank"][a:a + len(blk)], tf["t_gap"][a:a + len(blk)], tf["t_ncand"][a:a + len(blk)] = blk.T
            unc = (tf["t_rank"] == 0) & (tf["t_ncand"] == 1)
            out[env] = {}
            runs = [("A baseline @0.69", TD / f"p_{env}_A.npy", 0.69),
                    (f"B orphan-aware @{sel['threshold_B']:.2f} (A2-selected)", self.p(f"p_{env}_B.npy"), sel["threshold_B"]),
                    ("B orphan-aware @0.69", self.p(f"p_{env}_B.npy"), 0.69)]
            for name, pp, th in runs:
                if name.endswith("@0.69") and name.startswith("B") and abs(sel["threshold_B"] - 0.69) < 1e-9:
                    continue
                p = np.load(pp)
                keep = M.select(codes, p, th, True)
                _, parts = M.entity_f05(codes[keep], true, nq, return_parts=True)
                F, P, R, npred = parts["F"], parts["P"], parts["R"], parts["npred"]
                tp = int((keep & y).sum())
                fpm = keep & ~y
                single = ntrue == 0
                hist = np.histogram(p, bins=PBINS)[0] / len(p)
                out[env][name] = {
                    "threshold": th, "macro_f05": float(F.mean()), "pair_precision": tp / int(keep.sum()),
                    "pair_recall": tp / len(true), "entity_precision": float(P[npred > 0].mean()),
                    "entity_recall": float(R[ntrue > 0].mean()), "singleton_f05": float(F[single].mean()),
                    "singleton_false_merge_rate": float((npred[single] > 0).mean()),
                    "predictions": int(keep.sum()), "tp": tp, "fp": int(fpm.sum()), "fn": int(len(true) - tp),
                    "fp_on_singleton_s1": int((fpm & (ntrue[q] == 0)).sum()),
                    "fp_validation_pool": int((fpm & ~fit).sum()), "fp_fit_target_orphan": int((fpm & fit & tgt).sum()),
                    "fp_fit_distractor": int((fpm & fit & ~tgt).sum()),
                    "fp_t_rank0_uncontested": int((fpm & unc).sum()),
                    "precision_t_rank0": float(y[keep & (tf["t_rank"] == 0)].mean()),
                    "precision_uncontested": float(y[keep & unc].mean()) if (keep & unc).any() else None,
                    "fp_prob_ge_0.95": int((fpm & (p >= 0.95)).sum()),
                    "share_fp_prob_ge_0.95": float((p[fpm] >= 0.95).mean()), "mean_prob_fp": float(p[fpm].mean()),
                    "mean_prob_tp": float(p[keep & y].mean()),
                    "prob_hist_candidates": {f"[{a},{b})": float(h) for a, b, h in zip(PBINS[:-1], PBINS[1:], hist)},
                    "fp_t_features": {nm: {"mean": float(v[fpm].mean()), "share_zero": float((v[fpm] == 0).mean())}
                                      for nm, v in tf.items()},
                    "tp_t_features": {nm: {"mean": float(v[keep & y].mean()), "share_zero": float((v[keep & y] == 0).mean())}
                                      for nm, v in tf.items()}}
                r = out[env][name]
                log(f"  {env} | {name}: F0.5 {r['macro_f05']:.5f} P {r['pair_precision']:.4f} R {r['pair_recall']:.4f} "
                    f"single {r['singleton_f05']:.4f} FP {r['fp']:,} FN {r['fn']:,} orphanFP {r['fp_fit_target_orphan']:,}")
            jdump(self.p("evaluate.json"), out)


PHASES = ["hashes_before", "setup", "candidates", "idf", "features", "train", "select", "predict", "evaluate",
          "hashes_after"]
PHASE_OUT = {"hashes_before": "hashes_before.json", "setup": "setup.json", "candidates": "candidates.json",
             "idf": "idf_B_addr.npz", "features": "y_B.npy", "train": "train.json", "select": "select.json",
             "predict": "predict.done", "evaluate": "evaluate.json", "hashes_after": None}


def main():
    global MD, TD, SD, FD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", default=str(ROOT_EXP / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT_EXP / "_eda_cache"))
    ap.add_argument("--feat-dir", default=str(ROOT_EXP / "blocking" / "features"))
    ap.add_argument("--blocking-config", default=str(ROOT_EXP / "blocking" / "blocking_config.json"))
    ap.add_argument("--out-dir", default=str(ROOT_EXP / "diagnostics" / "orphan_aware_training"))
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sm = ROOT_EXP / "_smoke"
        MD, TD, SD, FD = sm / "matching", sm / "diagnostics_trank", sm / "test_scale" / "stress", sm / "final"
        a.split_dir, a.cache_dir = str(sm / "validation_split"), str(sm / "_eda_cache")
        a.feat_dir, a.blocking_config = str(sm / "blocking" / "features"), str(sm / "blocking" / "blocking_config.json")
        a.out_dir = str(sm / "diagnostics_orphan_aware")
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    e = Exp(a)
    tp = e.p("timings.json")
    timings = jload(tp) if tp.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and e.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(e, ph)()
        if o and o.endswith(".done"):
            e.p(o).write_text("ok", encoding="utf-8")
        timings[ph] = round(time.time() - t0, 1)
        jdump(tp, timings)
        log(f"[{ph}] done ({timings[ph]}s) {mem()}")


if __name__ == "__main__":
    main()
