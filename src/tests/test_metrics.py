"""Hand-computed checks of the competition metric (problem statement p.6).

Expected values are derived by hand as exact fractions:
F0.5 = 1.25·P·R / (0.25·P + R).
"""

import random
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ber.metrics import entity_f05, entity_prf, macro_f05  # noqa: E402


def f05_exact(p, r):
    return Fr(5, 4) * p * r / (Fr(1, 4) * p + r)


class TestEntityF05(unittest.TestCase):
    def test_worked_example_from_problem_statement(self):
        # pred {S2-00047, S2-00193, S3-00812}, true {S2-00047, S3-00812}: P=2/3, R=1 -> 0.714
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        true = {"S2-00047", "S3-00812"}
        p, r, f = entity_prf(pred, true)
        self.assertAlmostEqual(p, 2 / 3, places=15)
        self.assertEqual(r, 1.0)
        self.assertAlmostEqual(f, float(Fr(5, 7)), places=15)   # 0.714285...
        self.assertEqual(round(f, 3), 0.714)                     # value printed in the PDF

    def test_perfect(self):
        self.assertEqual(entity_f05({"S2-1", "S3-2"}, {"S3-2", "S2-1"}), 1.0)

    def test_singleton_correct_empty_is_one(self):
        self.assertEqual(entity_prf(set(), set()), (None, None, 1.0))

    def test_singleton_false_merge_is_zero(self):
        self.assertEqual(entity_f05({"S2-9"}, set()), 0.0)
        self.assertEqual(entity_f05({"S2-9", "S3-8"}, set()), 0.0)

    def test_empty_prediction_for_nonsingleton_is_zero(self):
        p, r, f = entity_prf(set(), {"S2-1"})
        self.assertIsNone(p)       # 0/0: undefined
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)

    def test_no_overlap_is_zero(self):
        self.assertEqual(entity_prf({"S2-5"}, {"S2-1", "S3-1"}), (0.0, 0.0, 0.0))

    def test_precision_one_recall_half(self):
        self.assertAlmostEqual(entity_f05({"S2-1"}, {"S2-1", "S2-2"}), float(Fr(5, 6)), places=15)

    def test_precision_half_recall_one(self):
        self.assertAlmostEqual(entity_f05({"S2-1", "S2-2"}, {"S2-1"}), float(Fr(5, 9)), places=15)

    def test_precision_third_recall_half(self):
        f = entity_f05({"S2-1", "S3-7", "S3-8"}, {"S2-1", "S2-2"})
        self.assertAlmostEqual(f, float(f05_exact(Fr(1, 3), Fr(1, 2))), places=15)   # 5/14
        self.assertAlmostEqual(f, 5 / 14, places=15)

    def test_precision_weighted_more_than_recall(self):
        self.assertGreater(entity_f05({"S2-1"}, {"S2-1", "S2-2"}),          # P=1, R=.5
                           entity_f05({"S2-1", "S2-2"}, {"S2-1"}))          # P=.5, R=1

    def test_order_and_container_irrelevant(self):
        self.assertEqual(entity_f05(["S3-2", "S2-1"], ("S2-1", "S3-2")), 1.0)

    def test_matches_general_fbeta_formula(self):
        rng = random.Random(0)
        universe = [f"S2-{i}" for i in range(12)]
        for _ in range(500):
            true = set(rng.sample(universe, rng.randint(1, 6)))
            pred = set(rng.sample(universe, rng.randint(1, 6)))
            tp = len(pred & true)
            if tp == 0:
                expected = 0.0
            else:
                p, r = Fr(tp, len(pred)), Fr(tp, len(true))
                expected = float((1 + Fr(1, 4)) * p * r / (Fr(1, 4) * p + r))
            self.assertAlmostEqual(entity_f05(pred, true), expected, places=14)


class TestMacroF05(unittest.TestCase):
    TRUTH = {
        "S1-1": {"S2-10", "S3-11"},                 # predicted perfectly           -> 1
        "S1-2": set(),                              # singleton, predicted empty    -> 1
        "S1-3": set(),                              # singleton, false merge        -> 0
        "S1-4": {"S2-40"},                          # non-singleton, predicted empty -> 0
        "S1-5": {"S2-00047", "S3-00812"},           # PDF worked example            -> 5/7
    }
    PRED = {
        "S1-1": {"S3-11", "S2-10"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": set(),
        "S1-5": {"S2-00047", "S2-00193", "S3-00812"},
    }

    def test_macro_average_hand_computed(self):
        rep = macro_f05(self.PRED, self.TRUTH)
        expected = (1 + 1 + 0 + 0 + Fr(5, 7)) / 5          # = 19/35
        self.assertAlmostEqual(rep.macro_f05, float(expected), places=15)
        self.assertEqual(rep.n_entities, 5)
        self.assertEqual(rep.n_true_singletons, 2)
        self.assertEqual(rep.n_true_nonsingletons, 3)
        self.assertEqual(rep.singletons_correct_empty, 1)
        self.assertEqual(rep.singletons_false_merge, 1)
        self.assertEqual(rep.nonsingletons_pred_empty, 1)
        self.assertEqual(rep.exact_set_matches, 2)
        self.assertAlmostEqual(rep.f05_singletons, 0.5, places=15)
        self.assertAlmostEqual(rep.f05_nonsingletons, float((1 + 0 + Fr(5, 7)) / 3), places=15)
        # diagnostics: tp=4, |pred|=6, |true|=5
        self.assertAlmostEqual(rep.micro_precision, 4 / 6, places=15)
        self.assertAlmostEqual(rep.micro_recall, 4 / 5, places=15)

    def test_macro_is_not_micro(self):
        truth = {"S1-1": {f"S2-{i}" for i in range(10)}, "S1-2": {"S3-1"}}
        pred = {"S1-1": {f"S2-{i}" for i in range(10)}, "S1-2": set()}
        rep = macro_f05(pred, truth)
        self.assertAlmostEqual(rep.macro_f05, 0.5, places=15)          # (1 + 0) / 2
        self.assertAlmostEqual(rep.micro_recall, 10 / 11, places=15)

    def test_all_empty_scores_singleton_fraction(self):
        truth = {"S1-1": set(), "S1-2": set(), "S1-3": {"S2-1"}, "S1-4": {"S3-1", "S3-2"}}
        rep = macro_f05({k: set() for k in truth}, truth)
        self.assertAlmostEqual(rep.macro_f05, 0.5, places=15)

    def test_all_singletons_all_empty_is_one(self):
        truth = {f"S1-{i}": set() for i in range(4)}
        self.assertEqual(macro_f05({k: set() for k in truth}, truth).macro_f05, 1.0)

    def test_missing_row_is_rejected(self):
        pred = dict(self.PRED)
        del pred["S1-2"]
        with self.assertRaises(ValueError):
            macro_f05(pred, self.TRUTH)

    def test_missing_row_diagnostic_mode_scores_empty(self):
        pred = dict(self.PRED)
        del pred["S1-2"]                  # singleton -> empty prediction -> 1.0, same total
        rep = macro_f05(pred, self.TRUTH, allow_missing=True)
        self.assertEqual(rep.missing_prediction_rows, 1)
        self.assertAlmostEqual(rep.macro_f05, float(Fr(19, 35)), places=15)

    def test_extra_row_is_rejected(self):
        pred = dict(self.PRED)
        pred["S1-999"] = set()
        with self.assertRaises(ValueError):
            macro_f05(pred, self.TRUTH)

    def test_groups(self):
        groups = {"S1-1": "US", "S1-2": "US", "S1-3": "India", "S1-4": "India", "S1-5": "India"}
        rep = macro_f05(self.PRED, self.TRUTH, groups=groups)
        self.assertAlmostEqual(rep.by_group["US"]["macro_f05"], 1.0, places=15)
        self.assertAlmostEqual(rep.by_group["India"]["macro_f05"], float(Fr(5, 7) / 3), places=15)


if __name__ == "__main__":
    unittest.main()
