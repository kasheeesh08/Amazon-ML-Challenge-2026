"""Small, deterministic histogram gradient-boosted decision trees (binary log-loss), in numba.

No GBDT library is available in this environment (no lightgbm/xgboost/sklearn) and nothing may be
downloaded, so this is a compact implementation of the standard algorithm:

* features are quantile-binned once (<= 64 bins, uint8, stored feature-major);
* each tree is grown depth-wise to ``max_depth``; per level one parallel pass builds (gradient, hessian)
  histograms for every active node, and the split maximising
  ``GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)`` is taken (``min_child_weight`` on hessians);
* leaf value = −G/(H+λ) × learning_rate; sample weights enter gradients and hessians;
* no row/feature subsampling — training is fully deterministic.
"""

import numpy as np
from numba import njit, prange


def make_bin_edges(X, max_bins=64, sample=400_000, seed=0):
    """Per-feature quantile edges (float32 arrays, strictly increasing) from a seeded row sample."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(n, sample), replace=False))
    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    edges = []
    for j in range(X.shape[1]):
        col = np.asarray(X[idx, j], dtype=np.float64)
        edges.append(np.unique(np.quantile(col, qs)).astype(np.float32))
    return edges


def apply_bins(X, edges, chunk=1_000_000):
    """Feature-major uint8 bins: bin = number of edges <= x (so ``bin <= thr`` ⇔ ``x < edges[thr]``)."""
    n, f = X.shape
    out = np.empty((f, n), dtype=np.uint8)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        blk = np.asarray(X[a:b], dtype=np.float32)
        for j in range(f):
            out[j, a:b] = np.searchsorted(edges[j], blk[:, j], side="right")
    return out


@njit(parallel=True, cache=True)
def _histograms(Bt, g, h, node_of_row, slot_of_node, n_slots, n_bins):
    F, n = Bt.shape
    H = np.zeros((F, n_slots, n_bins, 2))
    for f in prange(F):
        for i in range(n):
            s = slot_of_node[node_of_row[i]]
            if s < 0:
                continue
            b = Bt[f, i]
            H[f, s, b, 0] += g[i]
            H[f, s, b, 1] += h[i]
    return H


@njit(cache=True)
def _best_split(H, s, nb, lam, min_child):
    F = H.shape[0]
    best_gain, best_f, best_t = 0.0, -1, -1
    G = 0.0
    Hs = 0.0
    for b in range(H.shape[2]):
        G += H[0, s, b, 0]
        Hs += H[0, s, b, 1]
    parent = G * G / (Hs + lam)
    for f in range(F):
        GL = 0.0
        HL = 0.0
        for b in range(nb[f] - 1):
            GL += H[f, s, b, 0]
            HL += H[f, s, b, 1]
            GR = G - GL
            HR = Hs - HL
            if HL < min_child or HR < min_child:
                continue
            gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
            if gain > best_gain + 1e-12:
                best_gain, best_f, best_t = gain, f, b
    return best_gain, best_f, best_t, G, Hs


@njit(parallel=True, cache=True)
def _route(Bt, node_of_row, feat, thr, left, right):
    n = node_of_row.shape[0]
    for i in prange(n):
        nd = node_of_row[i]
        f = feat[nd]
        if f >= 0:
            node_of_row[i] = left[nd] if Bt[f, i] <= thr[nd] else right[nd]


@njit(cache=True)
def _leaf_sums(node_of_row, g, h, n_nodes):
    G = np.zeros(n_nodes)
    Hh = np.zeros(n_nodes)
    for i in range(node_of_row.shape[0]):
        G[node_of_row[i]] += g[i]
        Hh[node_of_row[i]] += h[i]
    return G, Hh


@njit(parallel=True, cache=True)
def _predict_tree(Bt, feat, thr, left, right, value, out):
    n = out.shape[0]
    for i in prange(n):
        nd = 0
        while feat[nd] >= 0:
            nd = left[nd] if Bt[feat[nd], i] <= thr[nd] else right[nd]
        out[i] += value[nd]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class GBDT:
    def __init__(self, n_trees=300, max_depth=6, learning_rate=0.1, lam=1.0, min_child_weight=20.0,
                 max_bins=64, seed=0):
        self.n_trees, self.max_depth, self.lr = n_trees, max_depth, learning_rate
        self.lam, self.min_child, self.max_bins, self.seed = lam, min_child_weight, max_bins, seed
        self.trees = []
        self.edges = None
        self.base = 0.0
        self.gain_importance = None

    def fit(self, X, y, w=None, log=None, feature_names=None):
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        self.edges = make_bin_edges(X, self.max_bins, seed=self.seed)
        Bt = apply_bins(X, self.edges)
        nb = np.array([len(e) + 1 for e in self.edges], dtype=np.int64)
        n_bins = int(nb.max())
        pbar = float(np.sum(w * y) / np.sum(w))
        self.base = float(np.log(pbar / (1 - pbar)))
        raw = np.full(len(y), self.base)
        self.gain_importance = np.zeros(X.shape[1])
        max_nodes = 2 ** (self.max_depth + 1)
        for it in range(self.n_trees):
            p = _sigmoid(raw)
            g = w * (p - y)
            h = w * np.maximum(p * (1 - p), 1e-12)
            feat = np.full(max_nodes, -1, dtype=np.int32)
            thr = np.zeros(max_nodes, dtype=np.int32)
            left = np.full(max_nodes, -1, dtype=np.int32)
            right = np.full(max_nodes, -1, dtype=np.int32)
            node_of_row = np.zeros(len(y), dtype=np.int32)
            active, n_nodes = [0], 1
            for depth in range(self.max_depth):
                if not active:
                    break
                slot = np.full(max_nodes, -1, dtype=np.int32)
                for s, nd in enumerate(active):
                    slot[nd] = s
                H = _histograms(Bt, g, h, node_of_row, slot, len(active), n_bins)
                nxt = []
                for s, nd in enumerate(active):
                    gain, f, t, _, _ = _best_split(H, s, nb, self.lam, self.min_child)
                    if f < 0:
                        continue
                    feat[nd], thr[nd] = f, t
                    left[nd], right[nd] = n_nodes, n_nodes + 1
                    nxt += [n_nodes, n_nodes + 1]
                    n_nodes += 2
                    self.gain_importance[f] += gain
                if nxt:
                    _route(Bt, node_of_row, feat, thr, left, right)
                    # nodes that split are internal now; only children are active next level
                active = nxt
            G, Hh = _leaf_sums(node_of_row, g, h, n_nodes)
            value = np.zeros(n_nodes)
            leaf = feat[:n_nodes] < 0
            value[leaf] = -G[leaf] / (Hh[leaf] + self.lam) * self.lr
            tree = (feat[:n_nodes].copy(), thr[:n_nodes].copy(), left[:n_nodes].copy(),
                    right[:n_nodes].copy(), value)
            self.trees.append(tree)
            _predict_tree(Bt, *tree, raw)
            if log and (it + 1) % 50 == 0:
                p = _sigmoid(raw)
                ll = -np.sum(w * (y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))) / np.sum(w)
                log(f"      tree {it + 1}: weighted train logloss {ll:.5f}")
        return self

    def raw_staged(self, X, stages, chunk=2_000_000):
        """Raw scores after each number of trees in ``stages`` (sorted), computed chunk-wise."""
        stages = sorted(stages)
        out = {s: np.empty(X.shape[0], dtype=np.float32) for s in stages}
        for a in range(0, X.shape[0], chunk):
            b = min(X.shape[0], a + chunk)
            Bt = apply_bins(X[a:b], self.edges)
            raw = np.full(b - a, self.base)
            k = 0
            for s in stages:
                while k < s:
                    _predict_tree(Bt, *self.trees[k], raw)
                    k += 1
                out[s][a:b] = raw
        return out

    def predict_proba(self, X, n_trees=None, chunk=2_000_000):
        n_trees = len(self.trees) if n_trees is None else n_trees
        return _sigmoid(self.raw_staged(X, [n_trees], chunk)[n_trees].astype(np.float64)).astype(np.float32)

    def save(self, path):
        arrs = {"base": np.array([self.base]), "n": np.array([len(self.trees)]),
                "params": np.array([self.max_depth, self.lr, self.lam, self.min_child, self.max_bins, self.seed]),
                "gain_importance": self.gain_importance}
        for j, e in enumerate(self.edges):
            arrs[f"edge_{j}"] = e
        arrs["n_features"] = np.array([len(self.edges)])
        for i, t in enumerate(self.trees):
            for name, a in zip(("feat", "thr", "left", "right", "value"), t):
                arrs[f"t{i}_{name}"] = a
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        p = z["params"]
        m = cls(n_trees=int(z["n"][0]), max_depth=int(p[0]), learning_rate=float(p[1]), lam=float(p[2]),
                min_child_weight=float(p[3]), max_bins=int(p[4]), seed=int(p[5]))
        m.base = float(z["base"][0])
        m.gain_importance = z["gain_importance"]
        m.edges = [z[f"edge_{j}"] for j in range(int(z["n_features"][0]))]
        m.trees = [tuple(z[f"t{i}_{nm}"] for nm in ("feat", "thr", "left", "right", "value"))
                   for i in range(int(z["n"][0]))]
        return m
