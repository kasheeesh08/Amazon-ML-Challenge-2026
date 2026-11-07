#!/usr/bin/env python3
"""Final competition inference pipeline (fixed configuration from Steps 2–3; nothing is tuned).

Configuration (verified by hash / read from the validated artifacts):
  * blocking   experiments/blocking/blocking_config.json (Step 2)
  * features   src/ber/match_features.py (Step 3, unchanged); IDF from the pool being matched
  * matcher    GBDT with matching_experiment.GBDT_PARAMS (300 trees, depth 6, lr 0.1, λ 1, min_child 20, seed 0)
  * decision   threshold 0.69, one-S1-per-target rule ON (experiments/matching/selection.json)

Final training data: ALL training S1 (2,206,821) blocked against the FULL training S2/S3 pool (10,320,219) with
the Step 2 configuration — every labelled pair can appear, every target's owner S1 is queried (no orphaned
targets), pool size ≈ test pool. Training rows = all positive candidate pairs + a deterministic 10 % sample of
negatives (weight 10), exactly the Step 3 sampling rule. Training rows are stored on disk (float32 memmap).

Test side: dataset/test is featurised and blocked with the same code; if the verified artifacts of
experiments/test_scale (same blocking config, same test files) exist they are reused (candidates, IDF,
target-context arrays), otherwise they are regenerated with experiments/test_scale_check.py.

Outputs: output/matching_results.tsv (submission), experiments/final/candidate_pairs.tsv (candidate set scored by
the matcher, for the final package), experiments/final/{FINAL_REPORT.md, reproducibility.json, ...}.

Phases: check, train_candidates, train_idf, train_cos, train_tc, train_rows, train_model, test_inputs, test_infer,
        decide, write, validate, report
"""

import argparse
import hashlib
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
from ber.features import FeatureStore  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.ids_tsv import CANDIDATE_HEADER, MATCHING_HEADER  # noqa: E402
from matching_experiment import GBDT_PARAMS, NEG_RATE  # noqa: E402  (the exact Step 3 settings)
from test_scale_check import chunks_by_query, dist, jdump, jload, peak_rss, sha256  # noqa: E402

PHASES = ["check", "train_candidates", "train_idf", "train_cos", "train_tc", "train_rows", "train_model",
          "test_inputs", "test_infer", "decide", "write", "validate", "report"]
VALIDATED = {"experiments/matching/model_gbdt.npz": "109324520a779ac8dcfcae8a1c422960c3a2fa2f3f3e9ed2b1b12aec0bd4c9e2",
             "experiments/matching/selection.json": "a5fcf91cf76b431c002e2add7821c8cce972712615227f12ce3ae3ebf2f15fc9",
             "experiments/blocking/blocking_config.json": "91a3897685414bb48c4bc4f00a162e1b245eb8640166f950a76bdc379042e0f5"}
N_TARGET_PARTS = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mem():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_working_set_gb": round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2),
                "private_gb": round(getattr(mi, "private", mi.rss) / 1e9, 2),
                "system_free_gb": round(psutil.virtual_memory().available / 1e9, 2)}
    except Exception:
        return {}


def iter_query_chunks(codes, chunk=2_000_000):
    """Query-aligned [a, b) chunks of a sorted (memory-mapped) code array — the SAME boundaries as
    test_scale_check.chunks_by_query, but found by reading small windows instead of materialising the
    whole array and its shifted copy (2 × 1.3 GB for the training candidates)."""
    n = len(codes)
    a = 0
    while a < n:
        b = min(n, a + chunk)
        if b < n:
            qb = int(codes[b]) >> B.Q_SHIFT
            w = 1024
            while True:                                   # window ending at b that contains qb's group start
                lo = max(a, b - w)
                seg = np.asarray(codes[lo:b]) >> B.Q_SHIFT
                if lo == a or seg[0] != qb:
                    break
                w *= 2
            b = lo + int(np.searchsorted(seg, qb, "left"))
            if b == a:                                    # chunk lies inside one query: extend to its end
                w = 1024
                while True:
                    hi = min(n, a + chunk + w)
                    seg = np.asarray(codes[a:hi]) >> B.Q_SHIFT
                    if hi == n or seg[-1] != qb:
                        break
                    w *= 2
                b = a + int(np.searchsorted(seg, qb, "right"))
        yield a, b
        a = b


class _SampleCachingMatrix:
    """Read-only view of the on-disk training matrix for GBDT.fit. ``make_bin_edges`` indexes it as
    X[idx, j] for every column j with the same idx; the sampled rows are read once and cached (identical
    values, one pass instead of 50 random-access passes). Slices pass straight through to the memmap."""

    def __init__(self, mm):
        self.mm = mm
        self.shape = mm.shape
        self._key = None
        self._rows = None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], np.ndarray):
            idx, j = key
            if self._key is None or not np.array_equal(self._key, idx):
                self._key, self._rows = idx.copy(), np.asarray(self.mm[idx])
            return self._rows[:, j]
        return self.mm[key]


def neg_sample(codes):
    """Step 3 rule (matching_experiment.Run.train_rows): keep a negative if hash % 1000 < NEG_RATE*1000."""
    with np.errstate(over="ignore"):
        h = ((np.asarray(codes).view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
    return h < np.uint64(int(NEG_RATE * 1000))


def target_context_partitioned(codes_path, cos_path, out_prefix, nrows2, nrows3, tmp_index):
    """Identical to MF.target_context over the whole set, computed per target-row partition."""
    codes = np.load(codes_path, mmap_mode="r")
    cos = np.load(cos_path, mmap_mode="r")
    n = len(codes)
    P = N_TARGET_PARTS

    def part_of(c):
        _, s, t = B.decode(c)
        return (s - 2) * P + np.minimum((t * P) // np.where(s == 2, nrows2, nrows3), P - 1)
    step = 20_000_000
    counts = np.zeros(2 * P, dtype=np.int64)
    for a in range(0, n, step):
        counts += np.bincount(part_of(np.asarray(codes[a:a + step])), minlength=2 * P)
    starts = np.concatenate([[0], np.cumsum(counts)])
    idx = np.lib.format.open_memmap(tmp_index, mode="w+", dtype=np.int64, shape=(n,))
    fill = starts[:-1].copy()
    for a in range(0, n, step):
        pid = part_of(np.asarray(codes[a:a + step]))
        order = np.argsort(pid, kind="stable")
        bnd = np.searchsorted(pid[order], np.arange(2 * P + 1))
        for k in range(2 * P):
            seg = order[bnd[k]:bnd[k + 1]] + a
            idx[fill[k]:fill[k] + len(seg)] = seg
            fill[k] += len(seg)
    outs = {nm: np.lib.format.open_memmap(f"{out_prefix}_{nm}.npy", mode="w+", dtype=np.float32, shape=(n,))
            for nm in ("t_rank", "t_gap", "t_ncand")}
    for k in range(2 * P):
        sel = np.sort(np.asarray(idx[starts[k]:starts[k + 1]]))
        if len(sel):
            r, g, s = MF.target_context(np.asarray(codes[sel]), np.asarray(cos[sel]))
            outs["t_rank"][sel], outs["t_gap"][sel], outs["t_ncand"][sel] = r, g, s
    for o in outs.values():
        o.flush()
    del idx, outs
    Path(tmp_index).unlink()


class Final:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sel = jload(ROOT / "experiments" / "matching" / "selection.json")["chosen"]
        self.spec = {k: self.sel[k] for k in ("model", "n_trees", "one_to_one", "best_threshold")}
        self.config = jload(a.blocking_config)
        self._ctx = None

    def p(self, n):
        return self.out / n

    @property
    def ctx(self):
        if self._ctx is None:
            from blocking_experiment import Context
            self._ctx = Context(self.a.train_feat_dir, self.a.split_dir, self.a.cache_dir)
        return self._ctx

    def test_stores(self):
        return {k: FeatureStore(Path(self.a.test_scale_dir) / "features_test", f"test_s{k}") for k in (1, 2, 3)}

    # ------------------------------------------------------------------------------------------
    def check(self):
        res = {"validated_hashes": {}, "spec": self.spec, "gbdt_params": GBDT_PARAMS, "neg_rate": NEG_RATE}
        if self.a.strict_hashes:
            for f, h in VALIDATED.items():
                cur = sha256(ROOT / f)
                res["validated_hashes"][f] = {"sha256": cur, "matches_validated": cur == h}
                if cur != h:
                    raise RuntimeError(f"{f} differs from the validated version")
        assert self.spec == {"model": "gbdt", "n_trees": 300, "one_to_one": True, "best_threshold": 0.69} or not self.a.strict_hashes
        assert GBDT_PARAMS["n_trees"] == self.spec["n_trees"]
        res["blocking_config_sha256"] = sha256(self.a.blocking_config)
        res["test_files"] = {f: os.stat(Path(self.a.test_dir) / f).st_size for f in sorted(os.listdir(self.a.test_dir))}
        res["test_dir_contains_only_sources"] = sorted(os.listdir(self.a.test_dir)) == \
            ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]
        jdump(self.p("check.json"), res)
        log(f"  check passed: {res}")

    # ---------------- final training environment: all train S1 vs full train pool -----------------
    def train_candidates(self):
        from blocking_experiment import lean_union
        ctx = self.ctx
        n1 = ctx.s[1].n
        pool = {k: np.arange(ctx.s[k].n) for k in (2, 3)}
        cdir = self.p("train_cand_chunks")
        cdir.mkdir(exist_ok=True)
        qc = self.a.query_chunk
        for i, a in enumerate(range(0, n1, qc)):
            f = cdir / f"c{i:03d}.npy"
            if f.exists():
                continue
            t0 = time.time()
            q = np.arange(a, min(n1, a + qc))
            u, info = lean_union(ctx, self.config, q, pool, ctx.truth(q))
            np.save(f, u + (np.int64(a) << np.int64(B.Q_SHIFT)))
            jdump(cdir / f"c{i:03d}.json", {"queries": int(len(q)), "pairs": int(len(u)),
                                           "seconds": round(time.time() - t0, 1), "memory": mem()})
            log(f"  train chunk {i}: {len(q):,} S1 -> {len(u):,} pairs ({time.time() - t0:.0f}s) {mem()}")
        files = sorted(cdir.glob("c*.npy"))
        total = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
        out = np.lib.format.open_memmap(self.p("train_cand.npy"), mode="w+", dtype=np.int64, shape=(total,))
        o = 0
        for f in files:
            c = np.load(f)
            out[o:o + len(c)] = c
            o += len(c)
        out.flush()
        del out
        true = ctx.truth(np.arange(n1))
        codes = np.load(self.p("train_cand.npy"), mmap_mode="r")
        hit = sum(int(np.isin(true, np.asarray(codes[a:a + 20_000_000])).sum()) for a in range(0, total, 20_000_000))
        jdump(self.p("train_candidates.json"), {"s1": int(n1), "pairs": int(total), "true_pairs": int(len(true)),
                                                "candidate_recall": hit / len(true),
                                                "chunks": [jload(f.with_suffix(".json")) for f in files]})
        for f in files:
            f.unlink()
        log(f"  train candidates {total:,}; candidate recall {hit / len(true):.5f}")

    def train_idf(self):
        ctx = self.ctx
        pool = {k: np.arange(ctx.s[k].n) for k in (2, 3)}
        for ch in ("name", "addr"):
            MF.build_idf({2: ctx.s[2], 3: ctx.s[3]}, pool, ch).save(self.p(f"train_idf_{ch}.npz"))

    def _train_X(self, codes, idf_n, idf_a):
        ctx = self.ctx
        return MF.pair_features(ctx.s[1], np.arange(ctx.s[1].n), {2: ctx.s[2], 3: ctx.s[3]}, codes, idf_n, idf_a)

    def train_cos(self):
        codes = np.load(self.p("train_cand.npy"), mmap_mode="r")
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"train_idf_{c}.npz")) for c in ("name", "addr"))
        cos = np.lib.format.open_memmap(self.p("train_cos.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        for a, b in iter_query_chunks(codes):
            cos[a:b] = self._train_X(np.asarray(codes[a:b]), idf_n, idf_a)[:, MF.COL["nameaddr_cos"]]
        cos.flush()

    def train_tc(self):
        target_context_partitioned(self.p("train_cand.npy"), self.p("train_cos.npy"), str(self.p("train")),
                                   self.ctx.s[2].n, self.ctx.s[3].n, self.p("_tc_index.npy"))

    def train_rows(self):
        """All positive candidate pairs + deterministic 10 % of negatives, streamed to an on-disk float32 matrix."""
        codes = np.load(self.p("train_cand.npy"), mmap_mode="r")
        tc = {nm: np.load(self.p(f"train_{nm}.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = (MF.IdfTable.load(self.p(f"train_idf_{c}.npz")) for c in ("name", "addr"))
        true = np.sort(self.ctx.truth(np.arange(self.ctx.s[1].n)))
        xf = open(self.p("train_X.f32"), "wb")
        ys, ws = [], []
        n_pos = n_neg_all = n_rows = 0
        for a, b in iter_query_chunks(codes):
            c = np.asarray(codes[a:b])
            y = np.isin(c, true, assume_unique=True)
            keep = y | neg_sample(c)
            X = self._train_X(c, idf_n, idf_a)
            for nm in tc:
                X[:, MF.COL[nm]] = tc[nm][a:b]
            X[keep].astype(np.float32).tofile(xf)
            xf.flush()
            os.fsync(xf.fileno())      # do not let GBs of dirty write-cache pages accumulate
            ys.append(y[keep].astype(np.uint8))
            ws.append(np.where(y[keep], 1.0, 1.0 / NEG_RATE).astype(np.float32))
            n_pos += int(y.sum())
            n_neg_all += int((~y).sum())
            n_rows += int(keep.sum())
        xf.close()
        np.save(self.p("train_y.npy"), np.concatenate(ys))
        np.save(self.p("train_w.npy"), np.concatenate(ws))
        jdump(self.p("train_rows.json"), {"candidate_pairs": int(len(codes)), "positive_pairs": n_pos,
                                          "negative_pairs": n_neg_all, "training_rows": n_rows,
                                          "training_positives": n_pos, "training_negatives": n_rows - n_pos,
                                          "neg_rate": NEG_RATE, "neg_weight": 1.0 / NEG_RATE,
                                          "n_features": MF.N_FEATURES})
        log(f"  training rows {n_rows:,} ({n_pos:,} positive, {n_rows - n_pos:,} sampled negatives)")

    def train_model(self):
        y = np.load(self.p("train_y.npy")).astype(np.float64)
        w = np.load(self.p("train_w.npy")).astype(np.float64)
        X = np.memmap(self.p("train_X.f32"), dtype=np.float32, mode="r", shape=(len(y), MF.N_FEATURES))
        t0 = time.time()
        gb = GBDT(**GBDT_PARAMS).fit(_SampleCachingMatrix(X), y, w, log=log)
        gb.save(self.p("model_final.npz"))
        jdump(self.p("train_model.json"), {"gbdt_params": GBDT_PARAMS, "rows": int(len(y)),
                                           "seconds": round(time.time() - t0, 1), "memory": mem(),
                                           "model_sha256": sha256(self.p("model_final.npz"))})
        log(f"  final GBDT trained ({time.time() - t0:.0f}s) {mem()}")

    # ---------------- test side ----------------------------------------------------------------
    def test_inputs(self):
        """Reuse verified test-scale artifacts (same blocking config + same test files) or regenerate them."""
        ts = Path(self.a.test_scale_dir)
        need = ["cand_test.npy", "idf_test_name.npz", "idf_test_addr.npz", "t_rank_test.npy", "t_gap_test.npy",
                "t_ncand_test.npy", "featurise.json", "candidates.json"]
        ok = all((ts / f).exists() for f in need)
        info = {"reused": False}
        if ok:
            meta = jload(ts / "featurise.json")
            stamps_ok = all(meta[str(k)]["stamp"]["size"] == os.stat(Path(self.a.test_dir) / f"test_source{k}.tsv").st_size
                            and meta[str(k)]["stamp"]["mtime"] == os.stat(Path(self.a.test_dir) / f"test_source{k}.tsv").st_mtime
                            for k in (1, 2, 3))
            ok = stamps_ok
        if not ok:
            log("  test-scale artifacts missing/stale -> regenerating with test_scale_check.py")
            cmd = [sys.executable, str(ROOT / "experiments" / "test_scale_check.py"), "--test-dir", str(self.a.test_dir),
                   "--blocking-config", str(self.a.blocking_config), "--out-dir", str(ts), "--workers", str(self.a.workers)]
            for ph in ("featurise", "candidates", "idf", "cos", "target_context"):
                subprocess.run(cmd + ["--phase", ph], check=True)
        else:
            info["reused"] = True
        codes = np.load(ts / "cand_test.npy", mmap_mode="r")
        S = self.test_stores()
        per_q = np.zeros(S[1].n, dtype=np.int64)
        for a in range(0, len(codes), 20_000_000):
            per_q += np.bincount(np.asarray(codes[a:a + 20_000_000]) >> B.Q_SHIFT, minlength=S[1].n)
        info.update({"test_s1": int(S[1].n), "pool_s2": int(S[2].n), "pool_s3": int(S[3].n),
                     "candidate_pairs": int(len(codes)), "candidates_per_s1": dist(per_q),
                     "s1_zero_candidates": int((per_q == 0).sum()),
                     "cand_test_sha256": sha256(ts / "cand_test.npy"),
                     "blocking_config_sha256": sha256(self.a.blocking_config)})
        jdump(self.p("test_inputs.json"), info)
        log(f"  test inputs: {info['candidate_pairs']:,} pairs for {info['test_s1']:,} S1 (reused={info['reused']})")

    def test_infer(self):
        ts = Path(self.a.test_scale_dir)
        S = self.test_stores()
        codes = np.load(ts / "cand_test.npy", mmap_mode="r")
        tc = {nm: np.load(ts / f"{nm}_test.npy", mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = (MF.IdfTable.load(ts / f"idf_test_{c}.npz") for c in ("name", "addr"))
        gb = GBDT.load(self.p("model_final.npz"))
        prob = np.lib.format.open_memmap(self.p("test_prob.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        t0 = time.time()
        for a, b in iter_query_chunks(codes):
            X = MF.pair_features(S[1], np.arange(S[1].n), {2: S[2], 3: S[3]}, np.asarray(codes[a:b]), idf_n, idf_a)
            for nm in tc:
                X[:, MF.COL[nm]] = tc[nm][a:b]
            prob[a:b] = gb.predict_proba(X, n_trees=self.spec["n_trees"])
        prob.flush()
        jdump(self.p("test_infer.json"), {"pairs_scored": int(len(codes)), "seconds": round(time.time() - t0, 1),
                                          "memory": mem()})

    def decide(self):
        ts = Path(self.a.test_scale_dir)
        S = self.test_stores()
        codes = np.load(ts / "cand_test.npy", mmap_mode="r")
        prob = np.load(self.p("test_prob.npy"), mmap_mode="r")
        th, one = self.spec["best_threshold"], self.spec["one_to_one"]
        idx = []
        for a in range(0, len(prob), 20_000_000):
            idx.append(np.flatnonzero(np.asarray(prob[a:a + 20_000_000]) >= th) + a)
        idx = np.concatenate(idx)
        c_pre, p_pre = np.asarray(codes[idx]), np.asarray(prob[idx])
        q, s, t = B.decode(c_pre)
        _, tcnt = np.unique((s.astype(np.int64) << 32) | t, return_counts=True)
        keep = M.select(c_pre, p_pre, th, one)
        pred = c_pre[keep]
        np.save(self.p("test_pred_codes.npy"), pred)
        qf, sf, _ = B.decode(pred)
        per = np.bincount(qf, minlength=S[1].n)
        ctry = np.asarray(S[1].col("country"))
        from ber.normalize import h64
        by_c = {c: {"s1": int((ctry == h64(c.casefold())).sum()),
                    "mean_predictions": float(per[ctry == h64(c.casefold())].mean()),
                    "zero_prediction_rate": float((per[ctry == h64(c.casefold())] == 0).mean())}
                for c in ("US", "India", "France") if (ctry == h64(c.casefold())).any()}
        res = {"threshold": th, "one_to_one": one, "pred_before_rule": int(len(idx)), "pred_final": int(len(pred)),
               "removed_by_rule": int(len(idx) - len(pred)), "targets_with_multiple_s1_before_rule": int((tcnt > 1).sum()),
               "pred_s2": int((sf == 2).sum()), "pred_s3": int((sf == 3).sum()),
               "predicted_positive_rate": len(pred) / len(prob),
               "s1_zero_predictions": int((per == 0).sum()), "s1_multiple_predictions": int((per >= 2).sum()),
               "predictions_per_s1": {"mean": float(per.mean()), "dist": {str(k): int(v) for k, v in enumerate(np.bincount(per))}},
               "by_country": by_c}
        jdump(self.p("decide.json"), res)
        log(f"  final predictions {len(pred):,}; zero-prediction S1 {res['s1_zero_predictions']:,}")

    # ---------------- outputs ------------------------------------------------------------------
    def _write_stream(self, path, header, codes_iter, q_ids, pool_ids):
        """Stream sorted unique codes (possibly in chunks) to a competition TSV with a row for every S1."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            nxt = 0
            for c in codes_iter:
                if len(c) == 0:
                    continue
                q, s, t = B.decode(c)
                bounds = np.flatnonzero(np.diff(q)) + 1
                starts = np.concatenate([[0], bounds])
                ends = np.concatenate([bounds, [len(q)]])
                for a, b in zip(starts, ends):
                    qq = int(q[a])
                    while nxt < qq:
                        f.write(f"{q_ids[nxt]}\t\n")
                        nxt += 1
                    ids = sorted(f"S{int(x)}-{int(pool_ids[int(x)][int(r)])}" for x, r in zip(s[a:b], t[a:b]))
                    f.write(f"{q_ids[qq]}\t{','.join(ids)}\n")
                    nxt = qq + 1
            while nxt < len(q_ids):
                f.write(f"{q_ids[nxt]}\t\n")
                nxt += 1

    def write(self):
        ts = Path(self.a.test_scale_dir)
        S = self.test_stores()
        q_ids = S[1].entity_ids()
        pool_ids = {2: np.asarray(S[2].col("id")), 3: np.asarray(S[3].col("id"))}
        pred = np.unique(np.load(self.p("test_pred_codes.npy")))
        sub = Path(self.a.submission_path)
        sub.parent.mkdir(parents=True, exist_ok=True)
        self._write_stream(sub, MATCHING_HEADER, [pred], q_ids, pool_ids)
        codes = np.load(ts / "cand_test.npy", mmap_mode="r")
        self._write_stream(self.p("candidate_pairs.tsv"), CANDIDATE_HEADER,
                           (np.asarray(codes[a:b]) for a, b in iter_query_chunks(codes)), q_ids, pool_ids)
        jdump(self.p("write.json"), {"matching_results": str(sub), "matching_results_sha256": sha256(sub),
                                     "matching_results_bytes": os.stat(sub).st_size,
                                     "candidate_pairs": str(self.p("candidate_pairs.tsv")),
                                     "candidate_pairs_bytes": os.stat(self.p("candidate_pairs.tsv")).st_size})
        log(f"  wrote {sub} and {self.p('candidate_pairs.tsv')}")

    def validate(self):
        sub = Path(self.a.submission_path)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        runs = {}
        for name, extra in (("matching_with_check_ids", ["--check-ids"]),):
            cmd = [sys.executable, str(ROOT / "utils" / "validate_submission.py"), "--matching", str(sub),
                   "--candidate", str(self.p("__skip_candidate__.tsv")), "--test-dir", str(self.a.test_dir)] + extra
            pr = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
            runs[name] = {"exit_code": pr.returncode, "stdout": pr.stdout.strip().splitlines()[-8:]}
            log(f"  official validator [{name}] exit={pr.returncode}: {pr.stdout.strip().splitlines()[-1]}")
        # candidate_pairs.tsv: the validator's candidate cross-check would load ~126M IDs into Python sets (its own
        # docstring advises dropping --candidate when memory is short), so the same rules are checked here by streaming.
        cand = self._check_candidates_streaming(self.p("candidate_pairs.tsv"), sub)
        jdump(self.p("validate.json"), {"official_validator": runs, "candidate_pairs_streaming_check": cand})
        if runs["matching_with_check_ids"]["exit_code"] != 0 or not cand["ok"]:
            raise RuntimeError("validation failed — see validate.json")

    def _check_candidates_streaming(self, cand_path, sub_path):
        from ber.ids_tsv import read_source_ids
        required = read_source_ids(Path(self.a.test_dir) / "test_source1.tsv")
        pool_nums = {k: np.sort(np.asarray(self.test_stores()[k].col("id"))) for k in (2, 3)}
        matched = {}
        with open(sub_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.partition("\t")
                ids = rest.rstrip("\n")
                if ids:
                    matched[s1] = set(ids.split(","))
        issues = {"header": None, "no_tab_rows": 0, "dup_rows": 0, "dup_ids_in_list": 0, "bad_prefix": 0,
                  "unknown_ids": 0, "missing_s1": 0, "extra_s1": 0, "matched_not_in_candidates": 0}
        seen = set()
        rows = 0
        with open(cand_path, encoding="utf-8") as f:
            issues["header"] = f.readline().rstrip("\n").split("\t") == list(CANDIDATE_HEADER)
            for line in f:
                s1, tab, rest = line.partition("\t")
                if not tab:
                    issues["no_tab_rows"] += 1
                    continue
                rows += 1
                if s1 in seen:
                    issues["dup_rows"] += 1
                seen.add(s1)
                ids = rest.rstrip("\n").split(",") if rest.strip() else []
                if len(ids) != len(set(ids)):
                    issues["dup_ids_in_list"] += 1
                for src in (2, 3):
                    nums = np.array([int(x[3:]) for x in ids if x.startswith(f"S{src}-")], dtype=np.int64)
                    if len(nums):
                        pos = np.minimum(np.searchsorted(pool_nums[src], nums), len(pool_nums[src]) - 1)
                        issues["unknown_ids"] += int((pool_nums[src][pos] != nums).sum())
                issues["bad_prefix"] += sum(1 for x in ids if not x.startswith(("S2-", "S3-")))
                if s1 in matched and not matched[s1] <= set(ids):
                    issues["matched_not_in_candidates"] += 1
        issues["missing_s1"] = len(required - seen)
        issues["extra_s1"] = len(seen - required)
        issues["rows"] = rows
        issues["ok"] = bool(issues["header"] and rows == len(required) and all(
            issues[k] == 0 for k in issues if k not in ("header", "rows")))
        log(f"  candidate_pairs.tsv streaming check: {issues}")
        return issues

    def report(self):
        write_report(self)


def write_report(fn):
    o = fn.out
    g = lambda n: jload(o / n) if (o / n).exists() else {}  # noqa: E731
    chk, tc, tr, tm, ti, inf, dc, wr, va = (g(n) for n in ("check.json", "train_candidates.json", "train_rows.json",
                                                         "train_model.json", "test_inputs.json", "test_infer.json",
                                                         "decide.json", "write.json", "validate.json"))
    ts_ref = jload(Path(fn.a.test_scale_dir) / "decide.json") if (Path(fn.a.test_scale_dir) / "decide.json").exists() else {}
    val_ref = jload(Path(fn.a.test_scale_dir) / "val_reference.json") if (Path(fn.a.test_scale_dir) / "val_reference.json").exists() else {}
    stress = jload(ROOT / "experiments" / "test_scale" / "stress" / "evaluate.json") if (ROOT / "experiments" / "test_scale" / "stress" / "evaluate.json").exists() else {}
    timings = g("timings.json")
    repro = {
        "pipeline": "experiments/final_pipeline.py",
        "decision": {"threshold": fn.spec["best_threshold"], "one_to_one_rule": fn.spec["one_to_one"],
                     "source": "experiments/matching/selection.json (Step 3, chosen on internal split A2)"},
        "model": {"type": "histogram GBDT (src/ber/gbdt.py)", "params": GBDT_PARAMS,
                  "n_trees_used": fn.spec["n_trees"], "file": str(o / "model_final.npz"),
                  "sha256": tm.get("model_sha256")},
        "training_data": {"description": "ALL training S1 blocked against the FULL training S2/S3 pool with the Step 2 "
                                         "configuration; all positive candidate pairs + deterministic 10% of negatives "
                                         "(weight 10), the Step 3 sampling rule",
                          "train_s1": tc.get("s1"), "candidate_pairs": tc.get("pairs"),
                          "candidate_recall": tc.get("candidate_recall"), **{k: tr.get(k) for k in (
                              "positive_pairs", "negative_pairs", "training_rows", "training_positives",
                              "training_negatives", "neg_rate", "neg_weight")}},
        "features": {"module": "src/ber/match_features.py", "n_features": MF.N_FEATURES, "names": MF.FEATURE_NAMES,
                     "idf": "from the pool being matched (training: full training pool; test: full test pool)"},
        "blocking": {"config": str(fn.a.blocking_config), "sha256": chk.get("blocking_config_sha256")},
        "validated_step3_hashes": chk.get("validated_hashes"),
        "test": ti, "inference": inf, "decisions": dc, "outputs": wr, "validation": va, "timings_s": timings,
    }
    jdump(o / "reproducibility.json", repro)
    L = []
    w = L.append
    w("# Final pipeline run report\n")
    w("Pipeline `experiments/final_pipeline.py`; fixed configuration from Steps 2–3 (nothing tuned). No test labels exist; "
      "no F0.5 is claimed for test.\n")
    w("## Configuration\n")
    w(f"- Blocking: `{fn.a.blocking_config}` (SHA-256 `{chk.get('blocking_config_sha256')}`) — Step 2, unchanged.")
    w(f"- Features: `src/ber/match_features.py`, {MF.N_FEATURES} features (Step 3, unchanged).")
    w(f"- Matcher: GBDT {GBDT_PARAMS}, {fn.spec['n_trees']} trees; decision threshold **{fn.spec['best_threshold']}**, "
      f"one-S1-per-target rule **{'ON' if fn.spec['one_to_one'] else 'OFF'}**. Final model SHA-256 `{tm.get('model_sha256')}`.")
    w(f"- Step 3 validated artifacts matched their recorded hashes: {chk.get('validated_hashes')}\n")
    w("## Final training data (training labels only)\n")
    w(f"- {tc.get('s1', 0):,} training S1 × full training pool → {tc.get('pairs', 0):,} candidate pairs "
      f"(candidate recall {100 * tc.get('candidate_recall', 0):.3f}%).")
    w(f"- Candidate pairs: {tr.get('positive_pairs', 0):,} positive / {tr.get('negative_pairs', 0):,} negative. Training rows: "
      f"{tr.get('training_rows', 0):,} = all {tr.get('training_positives', 0):,} positives + {tr.get('training_negatives', 0):,} "
      f"sampled negatives (rate {tr.get('neg_rate')}, weight {tr.get('neg_weight')}).")
    w(f"- GBDT training time {tm.get('seconds')}s; memory {tm.get('memory')}.\n")
    w("## Test candidates and inference\n")
    cps = ti.get("candidates_per_s1", {})
    w(f"- Test S1 {ti.get('test_s1', 0):,}; pool S2 {ti.get('pool_s2', 0):,} / S3 {ti.get('pool_s3', 0):,}; candidate pairs "
      f"{ti.get('candidate_pairs', 0):,}; candidates/S1 mean {cps.get('mean', 0):.1f}, median {cps.get('p50', 0):.0f}, "
      f"p95 {cps.get('p95', 0):.0f}, p99 {cps.get('p99', 0):.0f}, max {cps.get('max', 0):.0f}; S1 with zero candidates "
      f"{ti.get('s1_zero_candidates')}. Test-scale artifacts reused: {ti.get('reused')}.")
    w(f"- Inference: {inf.get('pairs_scored', 0):,} pairs scored in {inf.get('seconds')}s; memory {inf.get('memory')}.\n")
    w("## Final decisions (no labels — counts only)\n")
    w("| | validation (Step 3 model) | test, Step 3 model (diagnostic) | **test, final model** |")
    w("|---|---:|---:|---:|")
    rows = [("predicted pairs before rule", "pred_before_rule", "pred_before_rule", "pred_before_rule"),
            ("removed by one-to-one rule", "removed_by_one_to_one_rule", "removed_by_one_to_one_rule", "removed_by_rule"),
            ("final predicted pairs", "pred_after_rule", "pred_after_rule", "pred_final"),
            ("S1 with zero predictions", "s1_zero_predictions", "s1_zero_predictions", "s1_zero_predictions"),
            ("S1 with 2+ predictions", "s1_multiple_predictions", "s1_multiple_predictions", "s1_multiple_predictions"),
            ("predicted S2", "pred_s2", "pred_s2", "pred_s2"), ("predicted S3", "pred_s3", "pred_s3", "pred_s3")]
    for lab, kv, kt, kf in rows:
        w(f"| {lab} | {val_ref.get(kv, 0):,} | {ts_ref.get(kt, 0):,} | {dc.get(kf, 0):,} |")
    nv, nt = val_ref.get("n_s1", 1) or 1, dc and ti.get("test_s1", 1)
    w(f"| predictions per S1 | {val_ref.get('predictions_per_s1', {}).get('mean', 0):.3f} | "
      f"{ts_ref.get('predictions_per_s1', {}).get('mean', 0):.3f} | {dc.get('predictions_per_s1', {}).get('mean', 0):.3f} |")
    w(f"| zero-prediction rate | {100 * val_ref.get('s1_zero_prediction_rate', 0):.2f}% | {100 * ts_ref.get('s1_zero_prediction_rate', 0):.2f}% | "
      f"{100 * dc.get('s1_zero_predictions', 0) / max(1, ti.get('test_s1', 1)):.2f}% |")
    w("\nBy country (final): " + "; ".join(f"{c}: {d['mean_predictions']:.3f} predictions/S1, zero-prediction rate "
                                           f"{100 * d['zero_prediction_rate']:.2f}% (n={d['s1']:,})" for c, d in dc.get("by_country", {}).items()))
    if stress:
        w(f"\nLabelled stress reference (Step 3 model, validation S1 × full training pool): {stress['pair']['n_pred']:,} predictions "
          f"for 441,364 S1; macro F0.5 {stress['official_macro_f05']:.4f} (driven by orphaned targets, see experiments/TEST_SCALE_REPORT.md).")
    w("\n## Outputs and validation\n")
    w(f"- Submission: `{wr.get('matching_results')}` ({wr.get('matching_results_bytes', 0):,} bytes, SHA-256 `{wr.get('matching_results_sha256')}`).")
    w(f"- Candidate set scored by the matcher: `{wr.get('candidate_pairs')}` ({wr.get('candidate_pairs_bytes', 0):,} bytes) — needed "
      "as output/candidate_pairs.tsv in the final package.")
    for k, v in va.get("official_validator", {}).items():
        w(f"- Official validator ({k}): exit {v['exit_code']} — `{v['stdout'][-1] if v['stdout'] else ''}`")
    w(f"- candidate_pairs.tsv streaming check (validator rules + matched ⊆ candidates): {va.get('candidate_pairs_streaming_check')}")
    w(f"\nTimings (s): {timings}")
    (o / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  report: {o / 'FINAL_REPORT.md'}")


PHASE_OUT = {"check": "check.json", "train_candidates": "train_candidates.json", "train_idf": "train_idf_addr.npz",
             "train_cos": "train_cos.npy", "train_tc": "train_t_ncand.npy", "train_rows": "train_rows.json",
             "train_model": "train_model.json", "test_inputs": "test_inputs.json", "test_infer": "test_infer.json",
             "decide": "decide.json", "write": "write.json", "validate": "validate.json", "report": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-feat-dir", default=str(ROOT / "experiments" / "blocking" / "features"))
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_eda_cache"))
    ap.add_argument("--blocking-config", default=str(ROOT / "experiments" / "blocking" / "blocking_config.json"))
    ap.add_argument("--test-dir", default=str(ROOT / "dataset" / "test"))
    ap.add_argument("--test-scale-dir", default=str(ROOT / "experiments" / "test_scale"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "final"))
    ap.add_argument("--submission-path", default=str(ROOT / "output" / "matching_results.tsv"))
    ap.add_argument("--query-chunk", type=int, default=450_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-strict-hashes", dest="strict_hashes", action="store_false",
                    help="smoke tests only: skip the validated-hash assertions")
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    a = ap.parse_args()
    fn = Final(a)
    tp = fn.p("timings.json")
    timings = jload(tp) if tp.exists() else {}
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        o = PHASE_OUT[ph]
        if o and fn.p(o).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(fn, ph)()
        timings[ph] = round(time.time() - t0, 1)
        jdump(tp, timings)
        log(f"[{ph}] done ({timings[ph]}s) {mem()}")


if __name__ == "__main__":
    main()
