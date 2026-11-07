"""Matching decisions and metrics on candidate pairs (separate from blocking).

* ``select`` turns pair probabilities into predicted matches: probability >= threshold, optionally with
  a one-S1-per-target constraint (the training data has 0 S2/S3 records matched to more than one S1).
* ``entity_f05`` computes the competition's per-S1 F0.5 macro average vectorised over pair codes; it is
  cross-checked against ``ber.metrics.macro_f05`` (the Step 1 scorer) in tests and in the experiment.
* ``LogReg`` — weighted L2 logistic regression (scipy L-BFGS) used as the baseline model.
"""

import numpy as np
from scipy.optimize import minimize

from . import blocking as B


def select(codes, prob, threshold, one_to_one=False):
    """Boolean mask of predicted matches among ``codes`` (unique, sorted)."""
    m = np.asarray(prob) >= threshold
    if not one_to_one or not m.any():
        return m
    idx = np.flatnonzero(m)
    q, src, t = B.decode(np.asarray(codes)[idx])
    tkey = (src.astype(np.int64) << 32) | t
    order = np.lexsort((q, -np.asarray(prob)[idx], tkey))          # best S1 first within each target
    first = np.ones(len(order), dtype=bool)
    first[1:] = tkey[order][1:] != tkey[order][:-1]
    keep = np.zeros(len(m), dtype=bool)
    keep[idx[order[first]]] = True
    return keep


def entity_f05(pred_codes, true_codes, n_queries, return_parts=False):
    """Official per-entity F0.5 (β=0.5) macro-averaged over ALL ``n_queries`` S1 entities.

    true = ∅ & pred = ∅ → 1; true = ∅ & pred ≠ ∅ → 0; true ≠ ∅ & (pred = ∅ or no overlap) → 0."""
    pred = np.unique(np.asarray(pred_codes, dtype=np.int64))
    true = np.unique(np.asarray(true_codes, dtype=np.int64))
    pq = (pred >> B.Q_SHIFT).astype(np.int64)
    tq = (true >> B.Q_SHIFT).astype(np.int64)
    tp_mask = np.isin(pred, true, assume_unique=True)
    npred = np.bincount(pq, minlength=n_queries).astype(np.float64)
    ntrue = np.bincount(tq, minlength=n_queries).astype(np.float64)
    tp = np.bincount(pq[tp_mask], minlength=n_queries).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(ntrue > 0, tp / ntrue, 0.0)
        F = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    F = np.where(ntrue == 0, (npred == 0).astype(np.float64), F)
    if return_parts:
        return float(F.mean()), {"F": F, "P": P, "R": R, "npred": npred, "ntrue": ntrue, "tp": tp}
    return float(F.mean())


def pair_metrics(pred_codes, true_codes, blocked_true=None):
    """Pair-level precision / recall / F0.5 / F1. Recall is end-to-end (over ALL true pairs); if
    ``blocked_true`` (true pairs present in the candidates) is given, recall among those is added."""
    pred = np.unique(np.asarray(pred_codes, dtype=np.int64))
    true = np.unique(np.asarray(true_codes, dtype=np.int64))
    tp = int(np.isin(pred, true, assume_unique=True).sum())
    P = tp / len(pred) if len(pred) else 0.0
    R = tp / len(true) if len(true) else 0.0
    f = lambda b: (1 + b * b) * P * R / (b * b * P + R) if P + R > 0 else 0.0  # noqa: E731
    out = {"tp": tp, "n_pred": int(len(pred)), "n_true": int(len(true)), "precision": P, "recall": R,
           "f05": f(0.5), "f1": f(1.0)}
    if blocked_true is not None:
        out["recall_among_blocked"] = tp / len(blocked_true) if len(blocked_true) else 0.0
    return out


class LogReg:
    """Weighted L2 logistic regression on standardised features (float32 design, float64 optimiser)."""

    def __init__(self, l2=1.0, max_iter=300):
        self.l2, self.max_iter = l2, max_iter

    def fit(self, X, y, w):
        X = np.asarray(X, dtype=np.float32)
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sd = X.std(axis=0).astype(np.float32)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (X - self.mu) / self.sd
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        sw = w.sum()

        def fg(theta):
            b, coef = theta[0], theta[1:]
            z = Z @ coef.astype(np.float32) + b
            z = np.clip(z.astype(np.float64), -35, 35)
            p = 1 / (1 + np.exp(-z))
            loss = -np.sum(w * (y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))) / sw
            loss += 0.5 * self.l2 * np.dot(coef, coef) / sw
            r = (w * (p - y) / sw).astype(np.float32)
            g = np.empty_like(theta)
            g[0] = r.sum()
            g[1:] = (Z.T @ r).astype(np.float64) + self.l2 * coef / sw
            return loss, g

        res = minimize(fg, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.b, self.coef = float(res.x[0]), res.x[1:]
        self.converged = bool(res.success)
        return self

    def predict_proba(self, X, chunk=2_000_000):
        out = np.empty(X.shape[0], dtype=np.float32)
        for a in range(0, X.shape[0], chunk):
            Z = (np.asarray(X[a:a + chunk], dtype=np.float32) - self.mu) / self.sd
            z = np.clip(Z @ self.coef.astype(np.float32) + self.b, -35, 35)
            out[a:a + chunk] = 1 / (1 + np.exp(-z))
        return out

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, coef=self.coef, b=np.array([self.b]), l2=np.array([self.l2]))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(l2=float(z["l2"][0]))
        m.mu, m.sd, m.coef, m.b = z["mu"], z["sd"], z["coef"], float(z["b"][0])
        return m


def write_id_lists(path, header, codes, q_ids, pool_ids):
    """Stream pair codes to a competition-format TSV (one row per query, possibly empty, IDs sorted).

    ``header``: e.g. ``ids_tsv.MATCHING_HEADER``; ``q_ids``: S1 IDs by query position;
    ``pool_ids``: {2: numeric S2 id by file row, 3: numeric S3 id by file row}."""
    codes = np.unique(np.asarray(codes, dtype=np.int64))
    q, src, t = B.decode(codes)
    bounds = np.searchsorted(q, np.arange(len(q_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for i, s1 in enumerate(q_ids):
            a, b = bounds[i], bounds[i + 1]
            ids = sorted(f"S{int(s)}-{int(pool_ids[int(s)][int(r)])}" for s, r in zip(src[a:b], t[a:b]))
            f.write(f"{s1}\t{','.join(ids)}\n")
