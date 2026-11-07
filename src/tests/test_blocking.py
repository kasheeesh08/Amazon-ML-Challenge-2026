"""Tests for normalisation consistency with the EDA and for the blocking primitives."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ber import blocking as B  # noqa: E402
from ber import blocking_eval as E  # noqa: E402
from ber import normalize as N  # noqa: E402

SAMPLES = [
    ("Silver Anchor Marine PC", "446 Prospect Street, Lockport, NY", "US"),
    ("-- Horizon Nicos LLC", "440 LAKEVIEW DRIVE, MOBILE, AL", "US"),
    ("desertbsp.com", "3921 Pleasant Run Rd, Irving, Texas", "US"),
    ("सुपर कंसल्टेंसी प्राइवेट लिमिटेड", "H.NO 617 ROAD CIVIL LINES, NAGPUR, Maharashtra", "India"),
    ("Payne Énterprises", "0617 LINCOLN STREET, NEW BRITAIN, CT", "US"),
    ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
    ("", "", ""),
    ("Kris Mcclaran, O.D.", "MADRAS-600 017., CHENNAI", "India"),
]


class TestNormalizeMatchesEDA(unittest.TestCase):
    def test_identical_to_eda_module(self):
        import eda  # experiments/eda.py — the code that produced the EDA cache
        for name, addr, country in SAMPLES:
            a = N.normalize_record(name, addr, country)
            b = eda.normalize_record(name, addr, country)
            self.assertEqual(a, b)
            for c in N.KEY_COLS:
                self.assertEqual(N.h64(a[c]), eda.h64(b[c]))

    def test_blocking_tokens(self):
        r1 = N.normalize_record("Desert Bsp Inc", "3921 Pleasant Run Road, Irving, TX", "US")
        r2 = N.normalize_record("desertbsp.com", "3921 Pleasant Run Rd, Irving, Texas", "US")
        n1, a1 = N.blocking_tokens(r1, "Desert Bsp Inc")
        n2, a2 = N.blocking_tokens(r2, "desertbsp.com")
        self.assertNotIn("w:com", n2)               # TLD dropped for domain-like names
        self.assertIn("g:rtb", n1 & n2)             # trigram bridging "desert bsp" / "desertbsp"
        self.assertEqual(a1, a2)                    # Road/Rd and TX/Texas normalised
        self.assertIn("h:3921", a1)
        _, a3 = N.blocking_tokens(N.normalize_record("x", "0617 Lincoln St, New Britain, CT", "US"), "x")
        self.assertIn("h:617", a3)                  # leading zeros removed
        self.assertIn("d:617", a3)

    def test_flags(self):
        rec = N.normalize_record("-- desertbsp.com", "", "")
        f = N.record_flags("-- desertbsp.com", "", "", rec)
        for bit in (N.FLAG_ADDR_MISSING, N.FLAG_COUNTRY_MISSING, N.FLAG_NAME_DOMAIN, N.FLAG_NAME_JUNK_PREFIX):
            self.assertTrue(f & bit)
        self.assertFalse(f & N.FLAG_NAME_NONLATIN)


class TestBlockingPrimitives(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        q = np.array([0, 5, 2_000_000])
        s = np.array([2, 3, 3])
        t = np.array([0, 5_285_602, 123])
        q2, s2, t2 = B.decode(B.encode(q, s, t))
        self.assertTrue((q == q2).all() and (s == s2).all() and (t == t2).all())

    def test_gather_and_merge_csr(self):
        ptr = np.array([0, 2, 2, 5])
        tok = np.array([10, 11, 20, 21, 22])
        p, t = B.gather_csr(ptr, tok, np.array([2, 0]))
        self.assertEqual(p.tolist(), [0, 3, 5])
        self.assertEqual(t.tolist(), [20, 21, 22, 10, 11])
        p2, t2 = B.merge_csr((p, t), (np.array([0, 1, 1]), np.array([99])))
        self.assertEqual(p2.tolist(), [0, 4, 6])
        self.assertEqual(t2.tolist(), [20, 21, 22, 99, 10, 11])

    def test_retrieve_topk_ranks_by_cosine(self):
        # pool: doc0 {1,2,3}, doc1 {1,9}, doc2 {7,8}; query {1,2,3} -> doc0 first, doc1 second, doc2 never
        pool = (np.array([0, 3, 5, 7]), np.array([1, 2, 3, 1, 9, 7, 8], dtype=np.int64))
        q = (np.array([0, 3, 4]), np.array([1, 2, 3, 555], dtype=np.int64))    # 2nd query: unseen token
        docs, sc = B.retrieve_topk(q, pool, k=3, max_postings=10 ** 6)
        self.assertEqual(docs[0].tolist(), [0, 1, -1])
        self.assertGreater(sc[0, 0], sc[0, 1])
        self.assertEqual(docs[1].tolist(), [-1, -1, -1])

    def test_retrieve_budget_uses_rarest_first(self):
        # token 5 is in every doc (df=4), token 6 only in doc3 (df=1); budget 1 -> only token 6 used
        pool = (np.array([0, 1, 2, 3, 5]), np.array([5, 5, 5, 5, 6], dtype=np.int64))
        q = (np.array([0, 2]), np.array([5, 6], dtype=np.int64))
        docs, _ = B.retrieve_topk(q, pool, k=4, max_postings=1)
        self.assertEqual(docs[0].tolist(), [3, -1, -1, -1])

    def test_exact_channel_and_cap(self):
        qk = np.array([7, 8, 0])
        pk = np.array([7, 8, 8, 8, 0, 7])
        rows = np.array([100, 101, 102, 103, 104, 105])
        codes = B.exact_channel(qk, pk, rows, 2, cap=None)
        q, s, t = B.decode(codes)
        self.assertEqual(sorted(zip(q.tolist(), t.tolist())),
                         [(0, 100), (0, 105), (1, 101), (1, 102), (1, 103)])
        q, _, t = B.decode(B.exact_channel(qk, pk, rows, 2, cap=2))     # block of 8 has 3 -> dropped
        self.assertEqual(sorted(zip(q.tolist(), t.tolist())), [(0, 100), (0, 105)])

    def test_evaluate_hand_example(self):
        true = np.unique(B.encode([0, 0, 1], [2, 3, 2], [10, 20, 30]))      # q2 is a singleton
        cand = B.encode([0, 0, 1, 2, 2], [2, 2, 3, 3, 3], [10, 11, 99, 1, 2])
        res, hit = E.evaluate(cand, true, n_queries=3, pool_size={2: 50, 3: 50})
        self.assertAlmostEqual(res["candidate_recall"], 1 / 3)
        self.assertEqual(res["true_pairs_recovered"], 1)
        self.assertEqual(res["n_candidate_pairs"], 5)
        self.assertAlmostEqual(res["reduction_ratio"], 1 - 5 / 300)
        self.assertEqual(res["candidates_per_singleton_s1"]["mean"], 2.0)
        self.assertAlmostEqual(res["matched_s1_with_any_true_recovered"], 0.5)

    def test_recall_at_k(self):
        codes = B.encode([0, 0, 0], [2, 2, 2], [5, 6, 7])
        ranks = np.array([0, 1, 2], dtype=np.int16)
        true = np.unique(B.encode([0], [2], [7]))
        out, _ = E.recall_at_k(codes, ranks, true, 1, [1, 2, 3])
        self.assertEqual([out[k]["candidate_recall"] for k in ("1", "2", "3")], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
