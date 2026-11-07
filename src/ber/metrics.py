"""The competition metric: per-Source-1-entity F_beta (beta = 0.5), macro-averaged.

Specification (problem statement p.6, docs/COMPETITION_NOTES.md §4):

    F_0.5 = (1.25 * P * R) / (0.25 * P + R),   P = |pred ∩ true| / |pred|,  R = |pred ∩ true| / |true|

computed for every Source-1 entity and averaged over ALL Source-1 entities in the evaluation set.

Cases:
* true = ∅, pred = ∅            -> 1.0   (stated explicitly: correctly predicted singleton)
* true = ∅, pred ≠ ∅            -> 0.0   (stated explicitly: false merge on a singleton)
* true ≠ ∅, pred = ∅            -> 0.0   (P = 0/0 is undefined in the formula, but R = 0 and the formula
                                          is 0 for every defined P; skipping the entity is excluded by
                                          "averaged across all Source 1 entities". Documented convention.)
* true ≠ ∅, pred ≠ ∅, no overlap -> 0.0   (P = R = 0 gives 0/0; same limit/convention as above)
* otherwise                      -> the formula.

Predictions and truths are sets: order is irrelevant and a well-formed submission has no duplicates
(duplicates are a rejection, enforced by :mod:`ber.ids_tsv`, not silently de-duplicated here).
"""

import math
from dataclasses import dataclass, field

BETA = 0.5
_B2 = BETA * BETA  # 0.25


def entity_prf(pred, true):
    """Return ``(precision, recall, f05)`` for one Source-1 entity. P/R are None when undefined."""
    pred, true = set(pred), set(true)
    if not true:
        return (None, None, 1.0) if not pred else (0.0, None, 0.0)
    if not pred:
        return (None, 0.0, 0.0)
    tp = len(pred & true)
    p = tp / len(pred)
    r = tp / len(true)
    if tp == 0:
        return (p, r, 0.0)
    return (p, r, (1 + _B2) * p * r / (_B2 * p + r))


def entity_f05(pred, true):
    """Per-entity F0.5 exactly as specified (see module docstring for the edge cases)."""
    return entity_prf(pred, true)[2]


@dataclass
class ScoreReport:
    macro_f05: float
    n_entities: int
    n_true_singletons: int
    n_true_nonsingletons: int
    f05_singletons: float          # mean F0.5 over true singletons
    f05_nonsingletons: float       # mean F0.5 over entities with >=1 true match
    singletons_correct_empty: int  # true = ∅, pred = ∅
    singletons_false_merge: int    # true = ∅, pred ≠ ∅
    nonsingletons_pred_empty: int  # true ≠ ∅, pred = ∅
    nonsingletons_no_overlap: int  # true ≠ ∅, pred ≠ ∅, overlap = ∅
    exact_set_matches: int         # pred == true (includes correct singletons)
    mean_precision_defined: float  # over entities where P is defined (pred ≠ ∅)
    mean_recall_defined: float     # over entities where R is defined (true ≠ ∅)
    micro_precision: float         # diagnostic only — NOT the competition metric
    micro_recall: float            # diagnostic only — NOT the competition metric
    n_predicted_ids: int
    n_true_ids: int
    missing_prediction_rows: int = 0
    by_group: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(self.__dict__)


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else float("nan")


def macro_f05(predictions, truth, *, allow_missing=False, groups=None):
    """Macro-averaged F0.5 over every Source-1 entity in ``truth``.

    ``predictions`` / ``truth``: ``{source1_id: iterable_of_S2/S3_ids}``.
    A Source-1 entity absent from ``predictions`` is a rejection in the competition, so it raises
    unless ``allow_missing=True`` (then it is scored as an empty prediction, for diagnostics only).
    Predictions for S1 IDs that are not in ``truth`` always raise (validator: "not in the test set").
    ``groups``: optional ``{source1_id: label}`` (e.g. country) for per-group macro scores.
    """
    extra = set(predictions) - set(truth)
    if extra:
        raise ValueError(f"{len(extra)} predicted S1 IDs are not in the evaluation set, "
                         f"e.g. {sorted(extra)[:5]}")
    missing = [s1 for s1 in truth if s1 not in predictions]
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} S1 entities have no prediction row (a rejection), "
                         f"e.g. {sorted(missing)[:5]}")

    f_all, f_single, f_multi, p_def, r_def = [], [], [], [], []
    by_group_f = {}
    c = dict(single_ok=0, single_fm=0, multi_empty=0, multi_zero=0, exact=0)
    tp_total = pred_total = true_total = 0
    for s1, true_ids in truth.items():
        true_set = set(true_ids)
        pred_set = set(predictions.get(s1, ()))
        p, r, f = entity_prf(pred_set, true_set)
        f_all.append(f)
        if groups is not None:
            by_group_f.setdefault(groups.get(s1, "<unknown>"), []).append(f)
        if p is not None:
            p_def.append(p)
        if r is not None:
            r_def.append(r)
        tp = len(pred_set & true_set)
        tp_total += tp
        pred_total += len(pred_set)
        true_total += len(true_set)
        if pred_set == true_set:
            c["exact"] += 1
        if not true_set:
            f_single.append(f)
            c["single_ok" if not pred_set else "single_fm"] += 1
        else:
            f_multi.append(f)
            if not pred_set:
                c["multi_empty"] += 1
            elif tp == 0:
                c["multi_zero"] += 1

    return ScoreReport(
        macro_f05=_mean(f_all),
        n_entities=len(f_all),
        n_true_singletons=len(f_single),
        n_true_nonsingletons=len(f_multi),
        f05_singletons=_mean(f_single),
        f05_nonsingletons=_mean(f_multi),
        singletons_correct_empty=c["single_ok"],
        singletons_false_merge=c["single_fm"],
        nonsingletons_pred_empty=c["multi_empty"],
        nonsingletons_no_overlap=c["multi_zero"],
        exact_set_matches=c["exact"],
        mean_precision_defined=_mean(p_def),
        mean_recall_defined=_mean(r_def),
        micro_precision=tp_total / pred_total if pred_total else float("nan"),
        micro_recall=tp_total / true_total if true_total else float("nan"),
        n_predicted_ids=pred_total,
        n_true_ids=true_total,
        missing_prediction_rows=len(missing),
        by_group={g: {"n": len(v), "macro_f05": _mean(v)} for g, v in sorted(by_group_f.items())},
    )
