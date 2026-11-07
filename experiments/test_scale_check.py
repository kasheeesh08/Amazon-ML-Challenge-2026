#!/usr/bin/env python3
"""Test-scale compatibility check of the EXISTING Step 3 matcher (diagnostic only).

Runs the unchanged pipeline on dataset/test (no labels exist): Step 2 featurisation + blocking config,
Step 3 features + the saved GBDT (300 trees) + threshold 0.69 + one-S1-per-target rule, all read from
experiments/matching/selection.json. Nothing is retrained or tuned; no matching_results.tsv is written.
Validation reference statistics come from the saved Step 3 artifacts (X_val.npy, cand_val.npy, pval_chosen.npy).

Memory safety (results identical to a single un-chunked run):
  * candidates are generated per chunk of S1 queries (retrieval and exact joins are per-query independent);
  * target-context features (t_rank, t_gap, t_ncand) are computed per target-row partition (a target's
    group never spans partitions);
  * pair features are recomputed chunk-wise at inference instead of storing a ~25 GB matrix.

Phases (outputs checkpointed in --out-dir; completed phases are reused):
  featurise, candidates, idf, cos, target_context, infer, decide, val_reference, report
"""

import argparse
import hashlib
import json
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
from ber.features import FeatureStore, featurise_dir  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.normalize import h64  # noqa: E402

CHUNK_PAIRS = 2_000_000
N_TARGET_PARTS = 8                     # per source
SAMPLE_PER_MILLE = 10                  # 1 % deterministic pair sample for percentiles / PSI
PHASES = ["featurise", "candidates", "idf", "cos", "target_context", "infer", "decide", "val_reference", "report"]
SHIFT_FEATURES = ["t_rank", "t_gap", "t_ncand", "rank_src", "gap_src", "n_cand_src", "rank_all", "n_cand_all",
                  "name_cos", "name_jacc", "addr_cos", "addr_jacc", "nameaddr_cos", "addr_cont_t", "name_cont_t",
                  "eq_name_n2", "eq_name_n3", "eq_addr_n2", "eq_house", "eq_addr_digits", "eq_postal",
                  "house_both", "t_addr_missing", "t_name_nonlatin", "t_name_domain", "t_name_junk", "is_s3"]


def log(*a):
    print(*a, flush=True)


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


def jdump(p, obj):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(conv(obj), indent=1), encoding="utf-8")
    tmp.replace(p)


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def peak_rss():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_wset", mi.rss))
    except Exception:
        return None


def sample_mask(codes):
    with np.errstate(over="ignore"):
        h = ((np.asarray(codes).view(np.uint64) * np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(40)) % np.uint64(1000)
    return h < np.uint64(SAMPLE_PER_MILLE)


def chunks_by_query(codes):
    q = codes >> B.Q_SHIFT
    n, a = len(codes), 0
    while a < n:
        b = min(n, a + CHUNK_PAIRS)
        if b < n:
            b = int(np.searchsorted(q, q[b], "left"))
            if b == a:
                b = int(np.searchsorted(q, q[a], "right"))
        yield a, b
        a = b


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    q = np.percentile(x, [1, 5, 25, 50, 75, 90, 95, 99])
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
            **{f"p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 90, 95, 99], q)}}


def rank_buckets(r):
    r = np.asarray(r)
    edges = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 9)]
    return {f"{a}" if a == b else (f"{a}+" if b >= 10 ** 9 else f"{a}-{b}"): float(((r >= a) & (r <= b)).mean())
            for a, b in edges}


def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, np.float64), np.asarray(cur, np.float64)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    rc = np.bincount(np.searchsorted(edges, ref, "right"), minlength=len(edges) + 1) / len(ref)
    cc = np.bincount(np.searchsorted(edges, cur, "right"), minlength=len(edges) + 1) / len(cur)
    rc, cc = np.clip(rc, 1e-6, None), np.clip(cc, 1e-6, None)
    return float(np.sum((cc - rc) * np.log(cc / rc)))


class Stores:
    def __init__(self, s):
        self.s = s


class Check:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.feat = self.out / "features_test"
        sel = jload(Path(a.matching_dir) / "selection.json")["chosen"]
        self.spec = {k: sel[k] for k in ("model", "n_trees", "one_to_one", "best_threshold")}
        self.config = jload(a.blocking_config)

    def p(self, n):
        return self.out / n

    def stores(self):
        return {k: FeatureStore(self.feat, f"test_s{k}") for k in (1, 2, 3)}

    # ------------------------------------------------------------------------------------------
    def featurise(self):
        featurise_dir(self.a.test_dir, self.feat, workers=self.a.workers, stem="test", log=log)
        jdump(self.p("featurise.json"), {k: FeatureStore(self.feat, f"test_s{k}").meta for k in (1, 2, 3)})

    def candidates(self):
        from blocking_experiment import lean_union
        S = self.stores()
        n1 = S[1].n
        qc = self.a.query_chunk
        pool = {k: np.arange(S[k].n) for k in (2, 3)}
        cdir = self.p("cand_chunks")
        cdir.mkdir(exist_ok=True)
        info = {}
        for i, a in enumerate(range(0, n1, qc)):
            path = cdir / f"c{i:03d}.npy"
            if path.exists():
                log(f"  chunk {i}: reuse")
                continue
            t0 = time.time()
            q_rows = np.arange(a, min(n1, a + qc))
            union, ch = lean_union(Stores(S), self.config, q_rows, pool, np.empty(0, np.int64))
            np.save(path, union + (np.int64(a) << np.int64(B.Q_SHIFT)))
            info[i] = {"queries": int(len(q_rows)), "pairs": int(len(union)), "seconds": round(time.time() - t0, 1),
                       "channels": {n: v["n_pairs"] for n, v in ch.items()}}
            jdump(cdir / f"c{i:03d}.json", info[i])
            log(f"  chunk {i}: {len(q_rows):,} queries -> {len(union):,} pairs ({info[i]['seconds']}s, peak RSS "
                f"{(peak_rss() or 0) / 1e9:.1f} GB)")
        files = sorted(cdir.glob("c*.npy"))
        total = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
        out = np.lib.format.open_memmap(self.p("cand_test.npy"), mode="w+", dtype=np.int64, shape=(total,))
        o = 0
        for f in files:
            c = np.load(f)
            out[o:o + len(c)] = c
            o += len(c)
        out.flush()
        del out
        c = np.load(self.p("cand_test.npy"), mmap_mode="r")
        assert np.all(np.diff(c[:: max(1, len(c) // 1_000_000)]) > 0)
        chunk_info = [jload(f.with_suffix(".json")) for f in files]
        jdump(self.p("candidates.json"), {"pairs": int(total), "chunks": chunk_info, "query_chunk": qc})
        for f in files:
            f.unlink()
        log(f"  test candidates: {total:,}")

    def idf(self):
        S = self.stores()
        pool = {k: np.arange(S[k].n) for k in (2, 3)}
        for ch in ("name", "addr"):
            MF.build_idf({2: S[2], 3: S[3]}, pool, ch).save(self.p(f"idf_test_{ch}.npz"))

    def _features(self, S, codes, idf_n, idf_a):
        return MF.pair_features(S[1], np.arange(S[1].n), {2: S[2], 3: S[3]}, codes, idf_n, idf_a)

    def cos(self):
        S = self.stores()
        codes = np.load(self.p("cand_test.npy"), mmap_mode="r")
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_test_{c}.npz")) for c in ("name", "addr"))
        cos = np.lib.format.open_memmap(self.p("cos_test.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        for a, b in chunks_by_query(codes):
            cos[a:b] = self._features(S, np.asarray(codes[a:b]), idf_n, idf_a)[:, MF.COL["nameaddr_cos"]]
        cos.flush()

    def target_context(self):
        """Identical to MF.target_context, computed per target-row partition."""
        S = self.stores()
        codes = np.load(self.p("cand_test.npy"), mmap_mode="r")
        cos = np.load(self.p("cos_test.npy"), mmap_mode="r")
        n = len(codes)
        nrows = {2: S[2].n, 3: S[3].n}
        P = N_TARGET_PARTS

        def part_of(c):
            _, s, t = B.decode(c)
            return (s - 2) * P + np.minimum((t * P) // np.where(s == 2, nrows[2], nrows[3]), P - 1)
        counts = np.zeros(2 * P, dtype=np.int64)
        for a in range(0, n, 20_000_000):
            counts += np.bincount(part_of(np.asarray(codes[a:a + 20_000_000])), minlength=2 * P)
        starts = np.concatenate([[0], np.cumsum(counts)])
        idx = np.lib.format.open_memmap(self.p("_tc_index.npy"), mode="w+", dtype=np.int64, shape=(n,))
        fill = starts[:-1].copy()
        for a in range(0, n, 20_000_000):
            pid = part_of(np.asarray(codes[a:a + 20_000_000]))
            order = np.argsort(pid, kind="stable")
            ps = pid[order]
            bnd = np.searchsorted(ps, np.arange(2 * P + 1))
            for k in range(2 * P):
                seg = order[bnd[k]:bnd[k + 1]] + a
                idx[fill[k]:fill[k] + len(seg)] = seg
                fill[k] += len(seg)
        outs = {nm: np.lib.format.open_memmap(self.p(f"{nm}_test.npy"), mode="w+", dtype=np.float32, shape=(n,))
                for nm in ("t_rank", "t_gap", "t_ncand")}
        for k in range(2 * P):
            sel = np.sort(np.asarray(idx[starts[k]:starts[k + 1]]))
            if not len(sel):
                continue
            r, g, s = MF.target_context(np.asarray(codes[sel]), np.asarray(cos[sel]))
            outs["t_rank"][sel], outs["t_gap"][sel], outs["t_ncand"][sel] = r, g, s
        for o in outs.values():
            o.flush()
        del idx, outs
        self.p("_tc_index.npy").unlink()

    def infer(self):
        S = self.stores()
        codes = np.load(self.p("cand_test.npy"), mmap_mode="r")
        tc = {nm: np.load(self.p(f"{nm}_test.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"idf_test_{c}.npz")) for c in ("name", "addr"))
        model_path = Path(self.a.matching_dir) / "model_gbdt.npz"
        gb = GBDT.load(model_path)
        assert self.spec["model"] == "gbdt" and len(gb.trees) >= self.spec["n_trees"]
        n = len(codes)
        prob = np.lib.format.open_memmap(self.p("prob_test.npy"), mode="w+", dtype=np.float32, shape=(n,))
        fsum = np.zeros(MF.N_FEATURES)
        rank_src_counts = np.zeros(11, dtype=np.int64)
        samples, sample_codes = [], []
        t0 = time.time()
        for a, b in chunks_by_query(codes):
            c = np.asarray(codes[a:b])
            X = self._features(S, c, idf_n, idf_a)
            for nm in ("t_rank", "t_gap", "t_ncand"):
                X[:, MF.COL[nm]] = tc[nm][a:b]
            prob[a:b] = gb.predict_proba(X, n_trees=self.spec["n_trees"])
            fsum += X.sum(axis=0, dtype=np.float64)
            rank_src_counts += np.bincount(np.minimum(X[:, MF.COL["rank_src"]].astype(np.int64), 10), minlength=11)
            m = sample_mask(c)
            samples.append(X[m])
            sample_codes.append(c[m])
        prob.flush()
        np.save(self.p("feature_sample_test.npy"), np.concatenate(samples))
        np.save(self.p("feature_sample_test_codes.npy"), np.concatenate(sample_codes))
        jdump(self.p("infer.json"), {"pairs_scored": int(n), "seconds": round(time.time() - t0, 1),
                                     "feature_means": dict(zip(MF.FEATURE_NAMES, (fsum / n).tolist())),
                                     "rank_src_counts_0_to_10plus": rank_src_counts,
                                     "model_sha256": sha256(model_path), "spec": self.spec,
                                     "peak_rss_bytes": peak_rss()})

    # ------------------------------------------------------------------------------------------
    def _decision_stats(self, codes, prob, n_s1, s1_country, labels):
        th, one = self.spec["best_threshold"], self.spec["one_to_one"]
        pre = np.asarray(prob) >= th
        idx = np.flatnonzero(pre)
        c_pre = np.asarray(codes[idx]) if not isinstance(codes, np.ndarray) else codes[idx]
        q, s, t = B.decode(c_pre)
        tkey = (s.astype(np.int64) << 32) | t
        _, tcount = np.unique(tkey, return_counts=True)
        keep = M.select(c_pre, np.asarray(prob[idx]) if not isinstance(prob, np.ndarray) else prob[idx], th, one)
        cf = c_pre[keep]
        qf, sf, _ = B.decode(cf)
        per = np.bincount(qf, minlength=n_s1)
        by_c = {}
        for h, lab in labels.items():
            m = s1_country == h
            if m.any():
                by_c[lab] = {"s1": int(m.sum()), "mean_predictions_per_s1": float(per[m].mean()),
                             "zero_prediction_rate": float((per[m] == 0).mean())}
        ph = np.histogram(np.asarray(prob), bins=np.linspace(0, 1, 21))[0]
        return {
            "pairs": int(len(prob)), "pred_before_rule": int(len(idx)), "pred_after_rule": int(len(cf)),
            "removed_by_one_to_one_rule": int(len(idx) - len(cf)),
            "targets_with_multiple_s1_before_rule": int((tcount > 1).sum()),
            "pairs_on_multi_s1_targets_before_rule": int(tcount[tcount > 1].sum()),
            "predicted_positive_rate": float(len(cf) / len(prob)),
            "predictions_per_s1": {"mean": float(per.mean()), "dist": {str(k): int(v) for k, v in enumerate(np.bincount(per))}},
            "s1_zero_predictions": int((per == 0).sum()), "s1_zero_prediction_rate": float((per == 0).mean()),
            "s1_multiple_predictions": int((per >= 2).sum()), "s1_multiple_prediction_rate": float((per >= 2).mean()),
            "pred_s2": int((sf == 2).sum()), "pred_s3": int((sf == 3).sum()),
            "by_country": by_c, "prob_histogram_20bins": ph.tolist(),
        }

    def _labels(self, store):
        return {h64(c): c for c in ("US", "India", "France")} | {h64(c.casefold()): c for c in ("US", "India", "France")}

    def decide(self):
        S = self.stores()
        codes = np.load(self.p("cand_test.npy"))
        prob = np.load(self.p("prob_test.npy"))
        ctry = np.asarray(S[1].col("country"))
        per_q = np.bincount(codes >> B.Q_SHIFT, minlength=S[1].n)
        res = self._decision_stats(codes, prob, S[1].n, ctry, self._labels(S[1]))
        res["candidates_per_s1"] = dist(per_q) | {"zero": int((per_q == 0).sum())}
        tr = np.load(self.p("t_rank_test.npy"))
        res["t_rank"] = dist(tr) | {"buckets": rank_buckets(tr)}
        res["n_s1"] = int(S[1].n)
        res["pool"] = {"S2": int(S[2].n), "S3": int(S[3].n)}
        res["candidates_per_s1_by_country"] = {
            lab: dist(per_q[ctry == h]) for h, lab in self._labels(S[1]).items() if (ctry == h).any()}
        jdump(self.p("decide.json"), res)
        log(f"  test: predicted {res['pred_after_rule']:,} pairs; zero-prediction S1 {res['s1_zero_prediction_rate']:.4f}")

    def val_reference(self):
        md = Path(self.a.matching_dir)
        codes = np.load(md / "cand_val.npy")
        prob = np.load(md / "pval_chosen.npy")
        XV = np.load(md / "X_val.npy", mmap_mode="r")
        from blocking_experiment import Context
        ctx = Context(self.a.train_feat_dir, self.a.split_dir, self.a.cache_dir)
        qv = np.flatnonzero(ctx.fold[1] == 1)
        ctry = np.asarray(ctx.s[1].col("country"))[qv]
        res = self._decision_stats(codes, prob, len(qv), ctry, self._labels(ctx.s[1]))
        saved = np.load(md / "val_selected_mask.npy")
        res["matches_saved_step3_selection"] = bool(res["pred_after_rule"] == int(saved.sum()))
        per_q = np.bincount(codes >> B.Q_SHIFT, minlength=len(qv))
        res["candidates_per_s1"] = dist(per_q) | {"zero": int((per_q == 0).sum())}
        fsum = np.zeros(MF.N_FEATURES)
        rank_src_counts = np.zeros(11, dtype=np.int64)
        tr = np.empty(len(codes), dtype=np.float32)
        samples = []
        for a in range(0, len(codes), CHUNK_PAIRS):
            X = np.asarray(XV[a:a + CHUNK_PAIRS])
            fsum += X.sum(axis=0, dtype=np.float64)
            rank_src_counts += np.bincount(np.minimum(X[:, MF.COL["rank_src"]].astype(np.int64), 10), minlength=11)
            tr[a:a + len(X)] = X[:, MF.COL["t_rank"]]
            samples.append(X[sample_mask(codes[a:a + len(X)])])
        np.save(self.p("feature_sample_val.npy"), np.concatenate(samples))
        res["t_rank"] = dist(tr) | {"buckets": rank_buckets(tr)}
        res["feature_means"] = dict(zip(MF.FEATURE_NAMES, (fsum / len(codes)).tolist()))
        res["rank_src_counts_0_to_10plus"] = rank_src_counts
        res["n_s1"] = int(len(qv))
        res["pool"] = {"S2": int((ctx.fold[2] == 1).sum()), "S3": int((ctx.fold[3] == 1).sum())}
        jdump(self.p("val_reference.json"), res)
        log(f"  val reference done (matches saved Step 3 selection: {res['matches_saved_step3_selection']})")

    # ------------------------------------------------------------------------------------------
    def report(self):
        write_report(self)


def write_report(ck):
    out = ck.out
    T, V = jload(out / "decide.json"), jload(out / "val_reference.json")
    I, C = jload(out / "infer.json"), jload(out / "candidates.json")
    Fs = np.load(out / "feature_sample_test.npy")
    Fv = np.load(out / "feature_sample_val.npy")
    shift = {}
    for f in SHIFT_FEATURES:
        j = MF.COL[f]
        shift[f] = {"val_mean": V["feature_means"][f], "test_mean": I["feature_means"][f],
                    "val_p50": float(np.median(Fv[:, j])), "test_p50": float(np.median(Fs[:, j])),
                    "val_p90": float(np.percentile(Fv[:, j], 90)), "test_p90": float(np.percentile(Fs[:, j], 90)),
                    "psi": psi(Fv[:, j], Fs[:, j])}
    jdump(out / "shift.json", shift)
    L = []
    w = L.append
    w("# Test-scale compatibility check of the Step 3 matcher (diagnostic only)\n")
    w("Existing pipeline, unchanged: Step 2 featurisation + `experiments/blocking/blocking_config.json`; Step 3 features "
      f"+ `experiments/matching/model_gbdt.npz`; spec from `selection.json`: `{I['spec']}`. **No test labels exist or were "
      "used; nothing below is an accuracy measurement.** No matching_results.tsv or submission was written.\n")
    w("## 1. Candidate distribution\n")
    w("| | validation | test |")
    w("|---|---:|---:|")
    w(f"| S1 queries | {V['n_s1']:,} | {T['n_s1']:,} |")
    w(f"| pool S2 / S3 | {V['pool']['S2']:,} / {V['pool']['S3']:,} | {T['pool']['S2']:,} / {T['pool']['S3']:,} |")
    w(f"| S1 per pool record | {V['n_s1'] / (V['pool']['S2'] + V['pool']['S3']):.4f} | {T['n_s1'] / (T['pool']['S2'] + T['pool']['S3']):.4f} |")
    w(f"| candidate pairs | {V['pairs']:,} | {T['pairs']:,} |")
    for k in ("mean", "p50", "p90", "p95", "p99", "max"):
        w(f"| candidates/S1 {k} | {V['candidates_per_s1'][k]:.1f} | {T['candidates_per_s1'][k]:.1f} |")
    w(f"| S1 with zero candidates | {V['candidates_per_s1']['zero']:,} | {T['candidates_per_s1']['zero']:,} |")
    w("\nTest candidates per S1 by country: " + "; ".join(
        f"{c}: mean {d['mean']:.1f}, p50 {d['p50']:.0f}, p99 {d['p99']:.0f} (n={d['n']:,})" for c, d in T["candidates_per_s1_by_country"].items()) + "\n")
    w("## 2. t_rank distribution (rank of the pair's S1 among all S1 that retrieved the same target; 0 = best)\n")
    w("| | validation | test |")
    w("|---|---:|---:|")
    for k in ("mean", "p50", "p75", "p90", "p95", "p99", "max"):
        w(f"| {k} | {V['t_rank'][k]:.3f} | {T['t_rank'][k]:.3f} |")
    for bk in V["t_rank"]["buckets"]:
        w(f"| share t_rank = {bk} | {100 * V['t_rank']['buckets'][bk]:.2f}% | {100 * T['t_rank']['buckets'][bk]:.2f}% |")
    w(f"\nPSI(t_rank) = {shift['t_rank']['psi']:.4f}; PSI(t_ncand) = {shift['t_ncand']['psi']:.4f}; PSI(t_gap) = {shift['t_gap']['psi']:.4f} "
      "(PSI < 0.1 negligible, 0.1–0.25 moderate, > 0.25 major).\n")
    w("## 3. Matcher inference on test (no labels — counts only)\n")
    w("| | validation (reference) | test |")
    w("|---|---:|---:|")
    rows = [("pairs scored", "pairs"), ("predicted before one-to-one rule", "pred_before_rule"),
            ("removed by one-to-one rule", "removed_by_one_to_one_rule"),
            ("targets with >1 S1 before rule", "targets_with_multiple_s1_before_rule"),
            ("predicted positive (final)", "pred_after_rule"), ("S1 with zero predictions", "s1_zero_predictions"),
            ("S1 with 2+ predictions", "s1_multiple_predictions"), ("predicted S2 / S3", None)]
    for lab, k in rows:
        if k is None:
            w(f"| {lab} | {V['pred_s2']:,} / {V['pred_s3']:,} | {T['pred_s2']:,} / {T['pred_s3']:,} |")
        else:
            w(f"| {lab} | {V[k]:,} | {T[k]:,} |")
    w(f"| predicted-positive rate | {100 * V['predicted_positive_rate']:.3f}% | {100 * T['predicted_positive_rate']:.3f}% |")
    w(f"| predictions per S1 (mean) | {V['predictions_per_s1']['mean']:.3f} | {T['predictions_per_s1']['mean']:.3f} |")
    w(f"| S1 zero-prediction rate | {100 * V['s1_zero_prediction_rate']:.2f}% | {100 * T['s1_zero_prediction_rate']:.2f}% |")
    w(f"| S1 2+ prediction rate | {100 * V['s1_multiple_prediction_rate']:.2f}% | {100 * T['s1_multiple_prediction_rate']:.2f}% |")
    w("\nPredictions per S1 (count → S1): validation " + ", ".join(f"{k}:{v:,}" for k, v in list(V["predictions_per_s1"]["dist"].items())[:10])
      + "; test " + ", ".join(f"{k}:{v:,}" for k, v in list(T["predictions_per_s1"]["dist"].items())[:10]) + "\n")
    w("By country:\n")
    w("| country | set | S1 | mean predictions/S1 | zero-prediction rate |")
    w("|---|---|---:|---:|---:|")
    for nm, R in (("validation", V), ("test", T)):
        for c, d in R["by_country"].items():
            w(f"| {c} | {nm} | {d['s1']:,} | {d['mean_predictions_per_s1']:.3f} | {100 * d['zero_prediction_rate']:.2f}% |")
    w("\nValidation truth for reference (Step 1/3): 5.585% singletons, mean 3.461 true matches per S1.\n")
    w("Probability histogram (20 bins, share of pairs): validation " + ", ".join(f"{100 * x / V['pairs']:.1f}" for x in V["prob_histogram_20bins"])
      + " | test " + ", ".join(f"{100 * x / T['pairs']:.1f}" for x in T["prob_histogram_20bins"]) + "\n")
    w("## 4. Feature distribution shift (validation vs test)\n")
    w("Means over all pairs; medians / p90 / PSI on a deterministic 1% pair sample.\n")
    w("| feature | val mean | test mean | val p50 | test p50 | val p90 | test p90 | PSI |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f, d in sorted(shift.items(), key=lambda kv: -kv[1]["psi"]):
        flag = " **major**" if d["psi"] > 0.25 else (" moderate" if d["psi"] > 0.1 else "")
        w(f"| `{f}` | {d['val_mean']:.4f} | {d['test_mean']:.4f} | {d['val_p50']:.3f} | {d['test_p50']:.3f} | "
          f"{d['val_p90']:.3f} | {d['test_p90']:.3f} | {d['psi']:.4f}{flag} |")
    w("\n## 5. Memory / runtime\n")
    w(f"- Candidate generation: {len(C['chunks'])} query chunks of {C['query_chunk']:,} (" +
      ", ".join(f"{c['seconds']}s" for c in C["chunks"]) + ").")
    w(f"- Inference: {I['pairs_scored']:,} pairs in {I['seconds']}s; peak working set of the inference process "
      f"{(I['peak_rss_bytes'] or 0) / 1e9:.2f} GB. Phase timings: {jload(out / 'timings.json') if (out / 'timings.json').exists() else '—'}")
    w("\n## 6. Safety checks\n")
    w(f"- Model file SHA-256 at inference: `{I['model_sha256']}`; spec {I['spec']} read from selection.json (not tuned).")
    w(f"- Validation reference recomputed with the same code reproduces the saved Step 3 selection: {V['matches_saved_step3_selection']}.")
    w("- Test directory contains only test_source{1,2,3}.tsv — no labels exist; no external data; no matching_results.tsv written.")
    (out / "TEST_SCALE_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {out / 'TEST_SCALE_REPORT.md'}")


PHASE_OUT = {"featurise": "featurise.json", "candidates": "candidates.json", "idf": "idf_test_addr.npz",
             "cos": "cos_test.npy", "target_context": "t_ncand_test.npy", "infer": "infer.json",
             "decide": "decide.json", "val_reference": "val_reference.json", "report": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", default=str(ROOT / "dataset" / "test"))
    ap.add_argument("--matching-dir", default=str(ROOT / "experiments" / "matching"))
    ap.add_argument("--blocking-config", default=str(ROOT / "experiments" / "blocking" / "blocking_config.json"))
    ap.add_argument("--train-feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "test_scale"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--query-chunk", type=int, default=450_000)
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    a = ap.parse_args()
    ck = Check(a)
    tpath = ck.p("timings.json")
    timings = jload(tpath) if tpath.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and ck.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(ck, ph)()
        timings[ph] = round(time.time() - t0, 1)
        jdump(tpath, timings)
        log(f"[{ph}] done ({timings[ph]}s)")


if __name__ == "__main__":
    main()
