"""Tests for the Step 3 matcher: features, models, thresholding, metrics, duplicates, S2/S3 handling."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber import blocking as B  # noqa: E402
from ber import match_features as MF  # noqa: E402
from ber import matcher as M  # noqa: E402
from ber.features import FeatureStore, featurise_file  # noqa: E402
from ber.gbdt import GBDT  # noqa: E402
from ber.metrics import macro_f05  # noqa: E402

HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HDR)
        for r in rows:
            f.write("\t".join(r) + "\n")


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        _write(d / "s1.tsv", [("S1-1", "Silver Anchor Marine PC", "446 Prospect Street, Lockport, NY", "US"),
                              ("S1-2", "Blue Deli", "3427 Magruder Drive, Murfreesboro, TN", "US")])
        _write(d / "s2.tsv", [("S2-10", "Silver-Anchor Marine LLC", "446 PROSPECT ST, LOCKPORT, NY", "US"),
                              ("S2-11", "Completely Different Co", "1 Other Road, Elsewhere, CA", "US")])
        _write(d / "s3.tsv", [("S3-10", "silver anchor marine", "", "US"),
                              ("S3-11", "Deliblue.Com", "342 Magruder Dr, Murfreesboro, Tennessee", "US")])
        st = d / "store"
        q = lambda *a, **k: None  # noqa: E731
        for k in (1, 2, 3):
            featurise_file(d / f"s{k}.tsv", st, f"s{k}", f"S{k}-", workers=1, log=q)
        cls.s = {k: FeatureStore(st, f"s{k}") for k in (1, 2, 3)}
        pools = {2: np.arange(2), 3: np.arange(2)}
        cls.idf_n = MF.build_idf({2: cls.s[2], 3: cls.s[3]}, pools, "name")
        cls.idf_a = MF.build_idf({2: cls.s[2], 3: cls.s[3]}, pools, "addr")
        # q0 -> S2 row0 (true), S2 row1, S3 row0 (true, no address); q1 -> S3 row1
        cls.codes = np.unique(B.encode([0, 0, 0, 1], [2, 2, 3, 3], [0, 1, 0, 1]))
        cls.X = MF.pair_features(cls.s[1], np.array([0, 1]), {2: cls.s[2], 3: cls.s[3]}, cls.codes,
                                 cls.idf_n, cls.idf_a)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def row(self, q, src, t):
        return self.X[int(np.flatnonzero(self.codes == B.encode([q], [src], [t])[0])[0])]

    def test_shape_and_names(self):
        self.assertEqual(self.X.shape, (4, MF.N_FEATURES))
        self.assertEqual(len(set(MF.FEATURE_NAMES)), MF.N_FEATURES)
        self.assertTrue(np.isfinite(self.X).all())

    def test_source_flag(self):
        C = MF.COL
        self.assertEqual(self.row(0, 2, 0)[C["is_s3"]], 0)
        self.assertEqual(self.row(0, 3, 0)[C["is_s3"]], 1)

    def test_exact_and_similarity_features(self):
        C = MF.COL
        r = self.row(0, 2, 0)                      # legal suffix + punctuation + abbreviations differ
        self.assertEqual(r[C["eq_name_n2"]], 1)
        self.assertEqual(r[C["eq_name_n1"]], 0)
        self.assertEqual(r[C["eq_addr_n2"]], 1)
        self.assertEqual(r[C["eq_house"]], 1)
        self.assertAlmostEqual(float(r[C["name_cos"]]), 1.0, places=5)
        self.assertAlmostEqual(float(r[C["name_jacc"]]), 1.0, places=5)
        bad = self.row(0, 2, 1)
        self.assertLess(bad[C["name_cos"]], 0.2)
        self.assertEqual(bad[C["eq_name_n2"]], 0)

    def test_missing_address_flag(self):
        C = MF.COL
        r = self.row(0, 3, 0)
        self.assertEqual(r[C["t_addr_missing"]], 1)
        self.assertEqual(r[C["addr_nt"]], 0)
        self.assertEqual(r[C["addr_cos"]], 0)

    def test_context_ranks(self):
        C = MF.COL
        # among q0's two S2 candidates the true one ranks first
        self.assertEqual(self.row(0, 2, 0)[C["rank_src"]], 0)
        self.assertEqual(self.row(0, 2, 1)[C["rank_src"]], 1)
        self.assertEqual(self.row(0, 2, 0)[C["n_cand_src"]], 2)
        self.assertEqual(self.row(0, 3, 0)[C["n_cand_all"]], 3)

    def test_set_sims_hand_computed(self):
        keys, vals, n, _ = B._vocab_build(np.array([5, 6, 7], dtype=np.int64))
        idf = np.array([1.0, 2.0, 3.0], dtype=np.float32)          # ids in first-seen order: 5->1, 6->2, 7->3
        na, nb, ni, wa, wb, wi = MF.set_sims(np.array([5, 6], dtype=np.int64), np.array([6, 7, 9], dtype=np.int64),
                                             keys, vals, idf, np.float32(4.0))
        self.assertEqual((na, nb, ni), (2, 3, 1))
        self.assertAlmostEqual(wa, 1 + 4)
        self.assertAlmostEqual(wb, 4 + 9 + 16)                     # unseen token 9 -> idf 4
        self.assertAlmostEqual(wi, 4)

    def test_duplicate_candidates_give_identical_rows(self):
        dup = np.concatenate([self.codes, self.codes[:1]])
        order = np.argsort(dup, kind="stable")
        X2 = MF.pair_features(self.s[1], np.array([0, 1]), {2: self.s[2], 3: self.s[3]}, dup[order],
                              self.idf_n, self.idf_a)
        sims = [MF.COL[c] for c in ("name_cos", "addr_cos", "eq_name_n2", "is_s3")]
        self.assertTrue(np.array_equal(X2[0, sims], X2[1, sims]))

    def test_target_context(self):
        codes = np.unique(B.encode([0, 1, 2], [2, 2, 3], [7, 7, 7]))     # S2-row7 twice, S3-row7 once
        r, g, s = MF.target_context(codes, np.array([0.9, 0.5, 0.1]))
        self.assertEqual(s.tolist(), [2, 2, 1])                  # S2 row 7 and S3 row 7 are different targets
        self.assertEqual(r.tolist(), [0, 1, 0])


class TestModels(unittest.TestCase):
    def _data(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 5)).astype(np.float32)
        y = ((X[:, 0] + 0.5 * X[:, 1] > 0.3) ^ (rng.random(n) < 0.02)).astype(np.float64)
        return X, y

    def test_gbdt_learns_and_is_deterministic(self):
        X, y = self._data()
        m1 = GBDT(n_trees=30, max_depth=3, min_child_weight=1.0).fit(X, y)
        m2 = GBDT(n_trees=30, max_depth=3, min_child_weight=1.0).fit(X, y)
        p1, p2 = m1.predict_proba(X), m2.predict_proba(X)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertGreater(((p1 >= 0.5) == y).mean(), 0.93)
        with tempfile.TemporaryDirectory() as d:
            m1.save(Path(d) / "g.npz")
            self.assertTrue(np.array_equal(GBDT.load(Path(d) / "g.npz").predict_proba(X), p1))
        staged = m1.raw_staged(X, [10, 30])
        self.assertTrue(np.allclose(1 / (1 + np.exp(-staged[30])), p1, atol=1e-6))

    def test_gbdt_weights_shift_probabilities(self):
        X, y = self._data()
        base = GBDT(n_trees=5, max_depth=2).fit(X, y).predict_proba(X).mean()
        w = np.where(y == 0, 5.0, 1.0)
        weighted = GBDT(n_trees=5, max_depth=2).fit(X, y, w).predict_proba(X).mean()
        self.assertLess(weighted, base)

    def test_logreg_learns_and_is_deterministic(self):
        X, y = self._data()
        a = M.LogReg().fit(X, y, np.ones(len(y))).predict_proba(X)
        b = M.LogReg().fit(X, y, np.ones(len(y))).predict_proba(X)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(((a >= 0.5) == y).mean(), 0.9)


class TestDecisionsAndMetrics(unittest.TestCase):
    def test_threshold(self):
        codes = np.unique(B.encode([0, 0, 1], [2, 3, 2], [1, 1, 2]))
        prob = np.array([0.9, 0.4, 0.6])
        self.assertEqual(M.select(codes, prob, 0.5).tolist(), [True, False, True])
        self.assertEqual(M.select(codes, prob, 0.95).sum(), 0)

    def test_one_to_one_keeps_best_s1_per_target_and_separates_sources(self):
        # S2-row5 wanted by q0 (0.9) and q1 (0.7); S3-row5 by q1 only -> kept
        codes = np.unique(B.encode([0, 1, 1], [2, 2, 3], [5, 5, 5]))
        q, s, t = B.decode(codes)
        prob = np.where((q == 0) & (s == 2), 0.9, 0.7)
        keep = M.select(codes, prob, 0.5, one_to_one=True)
        kept = {(int(a), int(b)) for a, b, k in zip(q, s, keep) if k}
        self.assertEqual(kept, {(0, 2), (1, 3)})

    def test_empty_and_singletons(self):
        # 3 S1: q0, q1 singletons; q2 has one true match
        true = B.encode([2], [2], [4])
        # all empty: q0 = 1, q1 = 1, q2 (match missed) = 0
        self.assertAlmostEqual(M.entity_f05(np.empty(0, np.int64), true, 3), 2 / 3)
        # false merge on singleton q1: q0 = 1, q1 = 0, q2 = 0
        pred = B.encode([1], [3], [8])
        self.assertAlmostEqual(M.entity_f05(pred, true, 3), 1 / 3)
        self.assertEqual(M.entity_f05(np.empty(0, np.int64), np.empty(0, np.int64), 4), 1.0)

    def test_duplicates_ignored_in_metrics(self):
        true = B.encode([0, 0], [2, 3], [1, 2])
        pred = B.encode([0, 0, 0], [2, 2, 3], [1, 1, 2])            # duplicate prediction of (0, S2, 1)
        self.assertEqual(M.entity_f05(pred, true, 1), 1.0)
        self.assertEqual(M.pair_metrics(pred, true)["n_pred"], 2)

    def test_vectorised_f05_equals_step1_scorer(self):
        rng = random.Random(3)
        n = 300
        true, pred = [], []
        for q in range(n):
            ts = {(rng.choice((2, 3)), rng.randrange(40)) for _ in range(rng.choice((0, 0, 1, 2, 3, 5)))}
            ps = {(rng.choice((2, 3)), rng.randrange(40)) for _ in range(rng.choice((0, 1, 2, 4)))}
            ps |= set(list(ts)[:rng.randrange(len(ts) + 1)])
            true += [(q, s, t) for s, t in ts]
            pred += [(q, s, t) for s, t in ps]
        enc = lambda L: B.encode([a for a, _, _ in L], [b for _, b, _ in L], [c for _, _, c in L]) if L else np.empty(0, np.int64)  # noqa: E731
        truth = {q: {(s, t) for a, s, t in true if a == q} for q in range(n)}
        preds = {q: {(s, t) for a, s, t in pred if a == q} for q in range(n)}
        self.assertAlmostEqual(M.entity_f05(enc(pred), enc(true), n), macro_f05(preds, truth).macro_f05, places=12)

    def test_pair_metrics_by_source(self):
        true = B.encode([0, 0], [2, 3], [1, 1])
        pred = B.encode([0], [3], [1])
        _, ps, _ = B.decode(pred)
        m2 = M.pair_metrics(pred[ps == 2], true[B.decode(true)[1] == 2])
        m3 = M.pair_metrics(pred[ps == 3], true[B.decode(true)[1] == 3])
        self.assertEqual((m2["recall"], m3["recall"], m3["precision"]), (0.0, 1.0, 1.0))

    def test_write_id_lists_rows_for_every_query(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.tsv"
            M.write_id_lists(p, ["source1_entity_id", "matched_entity_ids"], B.encode([1, 1], [3, 2], [0, 1]),
                             ["S1-7", "S1-8", "S1-9"], {2: np.array([10, 11]), 3: np.array([20])})
            self.assertEqual(p.read_text(encoding="utf-8").splitlines(),
                             ["source1_entity_id\tmatched_entity_ids", "S1-7\t", "S1-8\tS2-11,S3-20", "S1-9\t"])


if __name__ == "__main__":
    unittest.main()
