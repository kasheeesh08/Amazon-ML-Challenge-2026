#!/usr/bin/env python3
"""End-to-end reproduction: data -> blocking -> matching -> output/{matching_results,candidate_pairs}.tsv

    python src/run_pipeline.py --train-dir <dataset>/train --test-dir <dataset>/test \
        --work-dir work --output-dir output

Phases (each checkpoints to --work-dir; an interrupted run resumes with the same command):
  featurise        train_source{1,2,3}.tsv and test_source{1,2,3}.tsv -> on-disk feature stores
  train_candidates Step 2 blocking of ALL training S1 against the FULL training S2/S3 pool
  train_idf/cos/tc IDF tables, name+address cosine, target-context features for the training candidates
  train_rows       all positive candidate pairs + deterministic 10 % of negatives (weight 10) -> on-disk matrix
  train_model      histogram GBDT (config/matcher_config.json)
  test_candidates  Step 2 blocking of all test S1 against the full test S2/S3 pool
  test_idf/cos/tc  as for training, on the test candidates
  test_infer       GBDT probabilities for every test candidate pair
  decide           threshold 0.69 + one-S1-per-target rule
  write            output/matching_results.tsv and output/candidate_pairs.tsv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.features import FeatureStore, featurise_file  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.ids_tsv import CANDIDATE_HEADER, MATCHING_HEADER  # noqa: E402
from ber.pipeline import (candidate_union, iter_query_chunks, jdump, jload, load_truth, log, memory,  # noqa: E402
                          sha256, target_context_partitioned)

PHASES = ["featurise", "train_candidates", "train_idf", "train_cos", "train_tc", "train_rows", "train_model",
          "test_candidates", "test_idf", "test_cos", "test_tc", "test_infer", "decide", "write"]


class _SampleCachingMatrix:
    """On-disk training matrix for GBDT.fit: the bin-edge sample rows are read once (identical values)."""

    def __init__(self, mm):
        self.mm, self.shape, self._key, self._rows = mm, mm.shape, None, None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], np.ndarray):
            idx, j = key
            if self._key is None or not np.array_equal(self._key, idx):
                self._key, self._rows = idx.copy(), np.asarray(self.mm[idx])
            return self._rows[:, j]
        return self.mm[key]


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.blocking = jload(a.blocking_config)
        self.mcfg = jload(a.matcher_config)
        self.gbdt_params = {k: v for k, v in self.mcfg["gbdt_params"].items()}

    def p(self, n):
        return self.work / n

    def stores(self, split):
        return {k: FeatureStore(self.work / "features", f"{split}_s{k}") for k in (1, 2, 3)}

    # ------------------------------------------------------------------------------------------
    def featurise(self):
        for split, d in (("train", self.a.train_dir), ("test", self.a.test_dir)):
            for k in (1, 2, 3):
                featurise_file(Path(d) / f"{split}_source{k}.tsv", self.work / "features", f"{split}_s{k}",
                               f"S{k}-", workers=self.a.workers, log=log)

    def _candidates(self, split):
        S = self.stores(split)
        n1 = S[1].n
        pool = {k: np.arange(S[k].n) for k in (2, 3)}
        cdir = self.p(f"{split}_cand_chunks")
        cdir.mkdir(exist_ok=True)
        qc = self.mcfg["query_chunk"]
        for i, a in enumerate(range(0, n1, qc)):
            f = cdir / f"c{i:03d}.npy"
            if f.exists():
                continue
            t0 = time.time()
            u, _ = candidate_union(S, self.blocking, np.arange(a, min(n1, a + qc)), pool)
            np.save(f, u + (np.int64(a) << np.int64(B.Q_SHIFT)))
            log(f"  {split} chunk {i}: {len(u):,} pairs ({time.time() - t0:.0f}s) {memory()}")
        files = sorted(cdir.glob("c*.npy"))
        total = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
        out = np.lib.format.open_memmap(self.p(f"{split}_cand.npy"), mode="w+", dtype=np.int64, shape=(total,))
        o = 0
        for f in files:
            c = np.load(f)
            out[o:o + len(c)] = c
            o += len(c)
        out.flush()
        del out
        for f in files:
            f.unlink()
        cdir.rmdir()
        jdump(self.p(f"{split}_candidates.json"), {"s1": int(n1), "pairs": int(total)})
        log(f"  {split} candidates: {total:,}")

    def _idf(self, split):
        S = self.stores(split)
        pool = {k: np.arange(S[k].n) for k in (2, 3)}
        for ch in ("name", "addr"):
            MF.build_idf({2: S[2], 3: S[3]}, pool, ch).save(self.p(f"{split}_idf_{ch}.npz"))

    def _X(self, split, codes, idf_n, idf_a):
        S = self.stores(split)
        return MF.pair_features(S[1], np.arange(S[1].n), {2: S[2], 3: S[3]}, codes, idf_n, idf_a)

    def _idfs(self, split):
        return tuple(MF.IdfTable.load(self.p(f"{split}_idf_{c}.npz")) for c in ("name", "addr"))

    def _cos(self, split):
        codes = np.load(self.p(f"{split}_cand.npy"), mmap_mode="r")
        idf_n, idf_a = self._idfs(split)
        cos = np.lib.format.open_memmap(self.p(f"{split}_cos.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        for a, b in iter_query_chunks(codes, self.mcfg["feature_chunk_pairs"]):
            cos[a:b] = self._X(split, np.asarray(codes[a:b]), idf_n, idf_a)[:, MF.COL["nameaddr_cos"]]
        cos.flush()

    def _tc(self, split):
        S = self.stores(split)
        target_context_partitioned(self.p(f"{split}_cand.npy"), self.p(f"{split}_cos.npy"), str(self.p(split)),
                                   S[2].n, S[3].n, self.p("_tc_index.npy"))

    def _full_X(self, split, codes, a, b, idf_n, idf_a, tc):
        X = self._X(split, np.asarray(codes[a:b]), idf_n, idf_a)
        for nm in tc:
            X[:, MF.COL[nm]] = tc[nm][a:b]
        return X

    def train_candidates(self):
        self._candidates("train")

    def train_idf(self):
        self._idf("train")

    def train_cos(self):
        self._cos("train")

    def train_tc(self):
        self._tc("train")

    def train_rows(self):
        codes = np.load(self.p("train_cand.npy"), mmap_mode="r")
        tc = {nm: np.load(self.p(f"train_{nm}.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = self._idfs("train")
        true = load_truth(Path(self.a.train_dir) / "train_ground_truth.tsv", self.stores("train"))
        rate = self.mcfg["negative_sampling_rate"]
        ys, ws = [], []
        n_pos = n_rows = 0
        with open(self.p("train_X.f32"), "wb") as xf:
            for a, b in iter_query_chunks(codes, self.mcfg["feature_chunk_pairs"]):
                c = np.asarray(codes[a:b])
                y = np.isin(c, true, assume_unique=True)
                with np.errstate(over="ignore"):
                    h = ((c.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(1000)
                keep = y | (h < np.uint64(int(rate * 1000)))
                X = self._full_X("train", codes, a, b, idf_n, idf_a, tc)
                X[keep].astype(np.float32).tofile(xf)
                xf.flush()
                os.fsync(xf.fileno())
                ys.append(y[keep].astype(np.uint8))
                ws.append(np.where(y[keep], 1.0, self.mcfg["negative_weight"]).astype(np.float32))
                n_pos += int(y.sum())
                n_rows += int(keep.sum())
        np.save(self.p("train_y.npy"), np.concatenate(ys))
        np.save(self.p("train_w.npy"), np.concatenate(ws))
        jdump(self.p("train_rows.json"), {"candidate_pairs": int(len(codes)), "true_pairs": int(len(true)),
                                          "training_rows": n_rows, "positives": n_pos, "negatives": n_rows - n_pos})
        log(f"  training rows {n_rows:,} ({n_pos:,} positive)")

    def train_model(self):
        y = np.load(self.p("train_y.npy")).astype(np.float64)
        w = np.load(self.p("train_w.npy")).astype(np.float64)
        X = np.memmap(self.p("train_X.f32"), dtype=np.float32, mode="r", shape=(len(y), MF.N_FEATURES))
        gb = GBDT(**self.gbdt_params).fit(_SampleCachingMatrix(X), y, w, log=log)
        gb.save(self.p("model.npz"))
        jdump(self.p("train_model.json"), {"rows": int(len(y)), "gbdt_params": self.gbdt_params,
                                           "model_sha256": sha256(self.p("model.npz")), "memory": memory()})

    def test_candidates(self):
        self._candidates("test")

    def test_idf(self):
        self._idf("test")

    def test_cos(self):
        self._cos("test")

    def test_tc(self):
        self._tc("test")

    def test_infer(self):
        codes = np.load(self.p("test_cand.npy"), mmap_mode="r")
        tc = {nm: np.load(self.p(f"test_{nm}.npy"), mmap_mode="r") for nm in ("t_rank", "t_gap", "t_ncand")}
        idf_n, idf_a = self._idfs("test")
        gb = GBDT.load(self.p("model.npz"))
        prob = np.lib.format.open_memmap(self.p("test_prob.npy"), mode="w+", dtype=np.float32, shape=(len(codes),))
        for a, b in iter_query_chunks(codes, self.mcfg["feature_chunk_pairs"]):
            prob[a:b] = gb.predict_proba(self._full_X("test", codes, a, b, idf_n, idf_a, tc),
                                         n_trees=self.mcfg["n_trees_used"])
        prob.flush()

    def decide(self):
        codes = np.load(self.p("test_cand.npy"), mmap_mode="r")
        prob = np.load(self.p("test_prob.npy"), mmap_mode="r")
        th = self.mcfg["threshold"]
        idx = np.concatenate([np.flatnonzero(np.asarray(prob[a:a + 20_000_000]) >= th) + a
                              for a in range(0, len(prob), 20_000_000)])
        c_pre, p_pre = np.asarray(codes[idx]), np.asarray(prob[idx])
        pred = c_pre[M.select(c_pre, p_pre, th, self.mcfg["one_to_one_rule"])]
        np.save(self.p("test_pred.npy"), pred)
        jdump(self.p("decide.json"), {"threshold": th, "one_to_one_rule": self.mcfg["one_to_one_rule"],
                                      "before_rule": int(len(idx)), "final": int(len(pred))})
        log(f"  predicted matches: {len(pred):,}")

    def _write(self, path, header, chunks, q_ids, pool_ids):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            nxt = 0
            for c in chunks:
                if len(c) == 0:
                    continue
                q, s, t = B.decode(c)
                bounds = np.flatnonzero(np.diff(q)) + 1
                for a, b in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(q)]])):
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
        S = self.stores("test")
        q_ids = S[1].entity_ids()
        pool_ids = {2: np.asarray(S[2].col("id")), 3: np.asarray(S[3].col("id"))}
        out = Path(self.a.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._write(out / "matching_results.tsv", MATCHING_HEADER, [np.unique(np.load(self.p("test_pred.npy")))],
                    q_ids, pool_ids)
        codes = np.load(self.p("test_cand.npy"), mmap_mode="r")
        self._write(out / "candidate_pairs.tsv", CANDIDATE_HEADER,
                    (np.asarray(codes[a:b]) for a, b in iter_query_chunks(codes, self.mcfg["feature_chunk_pairs"])),
                    q_ids, pool_ids)
        log(f"  wrote {out / 'matching_results.tsv'} and {out / 'candidate_pairs.tsv'}")


PHASE_DONE = {"featurise": None, "train_candidates": "train_candidates.json", "train_idf": "train_idf_addr.npz",
              "train_cos": "train_cos.npy", "train_tc": "train_t_ncand.npy", "train_rows": "train_rows.json",
              "train_model": "train_model.json", "test_candidates": "test_candidates.json",
              "test_idf": "test_idf_addr.npz", "test_cos": "test_cos.npy", "test_tc": "test_t_ncand.npy",
              "test_infer": "test_prob.npy", "decide": "decide.json", "write": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", required=True, help="folder with train_source{1,2,3}.tsv and train_ground_truth.tsv")
    ap.add_argument("--test-dir", required=True, help="folder with test_source{1,2,3}.tsv")
    ap.add_argument("--work-dir", default="work", help="intermediate artifacts (~40 GB for the full data)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--blocking-config", default=str(HERE / "config" / "blocking_config.json"))
    ap.add_argument("--matcher-config", default=str(HERE / "config" / "matcher_config.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phase", default="all", choices=["all"] + PHASES)
    a = ap.parse_args()
    pl = Pipeline(a)
    for ph in PHASES:
        if a.phase not in ("all", ph):
            continue
        done = PHASE_DONE[ph]
        if done and pl.p(done).exists() and a.phase == "all":
            log(f"[{ph}] reuse")
            continue
        log(f"[{ph}]")
        t0 = time.time()
        getattr(pl, ph)()
        log(f"[{ph}] done ({time.time() - t0:.0f}s) {memory()}")


if __name__ == "__main__":
    main()
