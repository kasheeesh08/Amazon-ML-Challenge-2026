"""Blocking evaluation: candidate recall, candidate-set size and reduction ratio.

candidate_recall = |true pairs ∩ candidate pairs| / |true pairs|   (pairs = (S1, S2/S3 record))

This is the recall CEILING of any downstream matcher, not a model score. All quantities are computed
on int64 pair codes (see ``ber.blocking.encode``) with sorted-array joins — no Python sets of pairs.
"""

import numpy as np

from .blocking import decode, encode


def truth_codes(q_s1_rows, n_s1_file, pair_s1row, pair_src, pair_trow):
    """Codes of all ground-truth pairs whose S1 is among the queries (query order = q_s1_rows)."""
    pos = np.full(n_s1_file, -1, dtype=np.int64)
    pos[np.asarray(q_s1_rows, dtype=np.int64)] = np.arange(len(q_s1_rows))
    q = pos[pair_s1row]
    m = q >= 0
    return np.unique(encode(q[m], pair_src[m], pair_trow[m]))


def _stats(counts):
    counts = np.asarray(counts)
    if len(counts) == 0:
        return {}
    q = np.percentile(counts, [50, 90, 95, 99])
    return {"mean": float(counts.mean()), "median": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3]), "max": int(counts.max()),
            "zero_candidates": int((counts == 0).sum())}


def evaluate(cand_codes, true_codes, n_queries, pool_size, same_country_space=None, pair_masks=None):
    """Metrics of a candidate set against the truth for ``n_queries`` S1 records.

    ``pool_size``: {2: n S2 records in pool, 3: n S3 records} — defines the Cartesian space.
    ``same_country_space``: optional number of same-country (S1, record) pairs.
    ``pair_masks``: optional {name: boolean mask over true_codes} for edge-case recall.
    """
    cand = np.unique(np.asarray(cand_codes, dtype=np.int64))
    true = np.asarray(true_codes, dtype=np.int64)
    hit = np.isin(true, cand, assume_unique=True)
    tq, tsrc, _ = decode(true)
    cq, csrc, _ = decode(cand)
    per_q = np.bincount(cq, minlength=n_queries)
    true_per_q = np.bincount(tq, minlength=n_queries)
    hit_per_q = np.bincount(tq[hit], minlength=n_queries)
    matched = true_per_q > 0
    cart = n_queries * (pool_size[2] + pool_size[3])
    res = {
        "n_queries": int(n_queries),
        "n_true_pairs": int(len(true)),
        "true_pairs_recovered": int(hit.sum()),
        "candidate_recall": float(hit.mean()) if len(true) else float("nan"),
        "recall_s2": float(hit[tsrc == 2].mean()) if (tsrc == 2).any() else float("nan"),
        "recall_s3": float(hit[tsrc == 3].mean()) if (tsrc == 3).any() else float("nan"),
        "matched_s1_with_all_true_recovered": float((hit_per_q[matched] == true_per_q[matched]).mean()),
        "matched_s1_with_any_true_recovered": float((hit_per_q[matched] > 0).mean()),
        "n_candidate_pairs": int(len(cand)),
        "n_candidate_pairs_s2": int((csrc == 2).sum()),
        "n_candidate_pairs_s3": int((csrc == 3).sum()),
        "candidates_per_s1": _stats(per_q),
        "candidates_per_singleton_s1": _stats(per_q[~matched]),
        "n_singleton_s1": int((~matched).sum()),
        "cartesian_pairs": int(cart),
        "reduction_ratio": 1.0 - len(cand) / cart if cart else float("nan"),
        "pair_completeness_x_reduction": None,
    }
    if same_country_space:
        res["same_country_pairs"] = int(same_country_space)
        res["reduction_ratio_vs_same_country"] = 1.0 - len(cand) / same_country_space
    if pair_masks:
        res["recall_by_case"] = {
            name: {"n_true_pairs": int(m.sum()),
                   "recall": float(hit[m].mean()) if m.any() else None}
            for name, m in pair_masks.items()}
    res.pop("pair_completeness_x_reduction")
    return res, hit


def recall_at_k(codes, ranks, true_codes, n_queries, ks):
    """For a retrieval channel run at k_max: recall and mean candidates/S1 when truncated to each k."""
    order = np.argsort(codes, kind="stable")
    codes_s, ranks_s = codes[order], ranks[order]
    pos = np.searchsorted(codes_s, true_codes)
    pos_c = np.minimum(pos, max(len(codes_s) - 1, 0))
    found = (len(codes_s) > 0) & (codes_s[pos_c] == true_codes) if len(codes_s) else np.zeros(len(true_codes), bool)
    true_rank = np.where(found, ranks_s[pos_c], np.iinfo(np.int16).max)
    out = {}
    for k in ks:
        n_c = int((ranks < k).sum())
        out[str(k)] = {"candidate_recall": float((true_rank < k).mean()) if len(true_codes) else float("nan"),
                       "n_candidate_pairs": n_c, "mean_candidates_per_s1": n_c / n_queries}
    return out, true_rank
