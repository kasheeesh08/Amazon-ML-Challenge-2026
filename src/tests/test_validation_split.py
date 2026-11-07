"""Tests for the S1-level stratified split: selection rule and an end-to-end build on synthetic data."""

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import validation_split as vs  # noqa: E402


class TestStratifiedSelect(unittest.TestCase):
    def test_exact_total_and_per_stratum_bounds(self):
        rng = np.random.default_rng(1)
        strata = rng.integers(0, 37, size=10_007)
        keys = rng.integers(0, 2 ** 63, size=len(strata), dtype=np.uint64)
        m = vs.stratified_select(strata, keys)
        self.assertEqual(int(m.sum()), round(len(strata) / 5))
        for s in np.unique(strata):
            n_s = int((strata == s).sum())
            v_s = int(m[strata == s].sum())
            self.assertIn(v_s, (n_s // 5, -(-n_s // 5)), f"stratum {s}: {v_s} of {n_s}")

    def test_selection_is_lowest_keys_within_stratum(self):
        strata = np.array([0] * 10 + [1] * 5)
        keys = np.arange(15, 0, -1).astype(np.uint64)       # descending keys
        m = vs.stratified_select(strata, keys)
        # stratum 0: 2 lowest keys (rows 8, 9); stratum 1: 1 lowest key (row 14)
        self.assertEqual(sorted(np.flatnonzero(m).tolist()), [8, 9, 14])

    def test_deterministic_and_input_order_independent(self):
        ids = [f"S1-{i}" for i in range(2000)]
        strata = np.array([hash_ % 7 for hash_ in range(2000)])
        keys = vs.seeded_hash(ids, seed=123)
        sel = {ids[i] for i in np.flatnonzero(vs.stratified_select(strata, keys))}
        perm = list(range(2000))
        random.Random(5).shuffle(perm)
        ids_p = [ids[i] for i in perm]
        keys_p = vs.seeded_hash(ids_p, seed=123)
        sel_p = {ids_p[i] for i in np.flatnonzero(vs.stratified_select(strata[perm], keys_p))}
        self.assertEqual(sel, sel_p)
        self.assertNotEqual(sel, {ids[i] for i in np.flatnonzero(
            vs.stratified_select(strata, vs.seeded_hash(ids, seed=124)))})

    def test_empty(self):
        self.assertEqual(len(vs.stratified_select(np.array([], dtype=np.int64), np.array([], dtype=np.uint64))), 0)


def _write_source(path, prefix, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i, country in rows:
            f.write(f"{prefix}{i}\tName {i}\t{i} Main Street, Town\t{country}\n")


class TestEndToEnd(unittest.TestCase):
    def test_build_and_verify_synthetic(self):
        rng = random.Random(7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, cache, out = tmp / "train", tmp / "cache", tmp / "out"
            data.mkdir()
            cache.mkdir()
            countries = ["US", "India"]
            s1 = [(1000 + i, countries[i % 2]) for i in range(200)]
            s2 = [(5000 + i, countries[i % 2]) for i in range(600)]
            s3 = [(9000 + i, countries[i % 2]) for i in range(600)]
            free2 = [i for i, _ in s2]
            free3 = [i for i, _ in s3]
            rng.shuffle(free2)
            rng.shuffle(free3)
            gt = {}
            for i, c in s1:
                k2 = rng.choice([0, 0, 1, 2, 3])
                k3 = rng.choice([0, 1, 2])
                # keep matches within the same country (as in the real data)
                t2 = [x for x in free2 if s2[x - 5000][1] == c][:k2]
                t3 = [x for x in free3 if s3[x - 9000][1] == c][:k3]
                for x in t2:
                    free2.remove(x)
                for x in t3:
                    free3.remove(x)
                gt[i] = [f"S2-{x}" for x in t2] + [f"S3-{x}" for x in t3]
            _write_source(data / "train_source1.tsv", "S1-", s1)
            _write_source(data / "train_source2.tsv", "S2-", s2)
            _write_source(data / "train_source3.tsv", "S3-", s3)
            with open(data / "train_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for i, _ in s1:
                    f.write(f"S1-{i}\t{','.join(gt[i])}\n")
            # minimal EDA-style cache
            chash = {"US": 11, "India": 22}
            for key, rows in (("train_s1", s1), ("train_s2", s2), ("train_s3", s3)):
                np.save(cache / f"{key}__id.npy", np.array([i for i, _ in rows], dtype=np.int64))
                np.save(cache / f"{key}__country.npy", np.array([chash[c] for _, c in rows], dtype=np.int64))
            pr, pt, ps = [], [], []
            for r, (i, _) in enumerate(s1):
                for t in gt[i]:
                    pr.append(r)
                    pt.append(int(t[3:]))
                    ps.append(int(t[1]))
            np.save(cache / "gt__s1.npy", np.array([i for i, _ in s1], dtype=np.int64))
            np.save(cache / "gt__pair_row.npy", np.array(pr, dtype=np.int32))
            np.save(cache / "gt__pair_t.npy", np.array(pt, dtype=np.int64))
            np.save(cache / "gt__pair_src.npy", np.array(ps, dtype=np.int8))
            for key, fname in (("train_s1", "train_source1.tsv"), ("train_s2", "train_source2.tsv"),
                               ("train_s3", "train_source3.tsv"), ("gt", "train_ground_truth.tsv")):
                st = os.stat(data / fname)
                (cache / f"{key}__stamp.json").write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime}))

            quiet = lambda *a, **k: None  # noqa: E731
            man = vs.build(data, cache, out, seed=99, log=quiet)
            ver = vs.verify(data, out, run_validator=True, log=quiet)
            self.assertTrue(ver["all_checks_passed"], ver)
            self.assertEqual(man["counts"]["val"]["s1_entities"], 40)
            self.assertEqual(ver["fit_targets_inside_val_pool"], 0)
            self.assertEqual(ver["official_validator_on_val_ground_truth"]["exit_code"], 0)
            # rebuilding with the same seed reproduces byte-identical outputs
            man2 = vs.build(data, cache, tmp / "out2", seed=99, log=quiet)
            self.assertEqual(man["outputs_sha256"], man2["outputs_sha256"])
            # the fit fold reconstructed from the assignment arrays is the exact complement
            val_ids = {line.split("\t")[0] for line in
                       (out / vs.VAL_DIR_NAME / "test_source1.tsv").read_text(encoding="utf-8").splitlines()[1:]}
            fit_ids = {line.split("\t")[0] for line in vs.iter_fold_lines("s1", 0, data, out)}
            self.assertEqual(len(val_ids & fit_ids), 0)
            self.assertEqual(len(val_ids | fit_ids), 200)


if __name__ == "__main__":
    unittest.main()
