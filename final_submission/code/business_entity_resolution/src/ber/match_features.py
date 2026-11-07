"""Pairwise features for the matcher, computed from the Step 2 feature store (no raw text needed).

Every candidate pair (S1 query, S2/S3 record) gets ``len(FEATURE_NAMES)`` float32 features:

* ``is_s3`` — keeps Source 2 and Source 3 distinguishable.
* ``eq_<key>`` — equality of each normalised exact key (``normalize.KEY_COLS``; 0 if either is empty),
  plus ``*_both`` presence indicators for postal code, house number and address digits.
* record flags (``normalize.FLAG_*``) of the target (missing name/address, non-Latin, domain-like, junk
  prefix, postal present, landmark) and of the query (postal present, landmark).
* set similarities on the hashed **name** tokens (words + character trigrams) and **address** tokens
  (words, digit groups, house number): sizes, intersection, Jaccard, containment both ways, IDF cosine,
  IDF-weighted Jaccard; ``nameaddr_cos`` over both. IDF = log((N+1)/(df+1)) + 1 from the candidate pool
  (same formula as the Step 2 retrieval), so train / validation / test use their own pool.
* candidate-set context (label-free): rank / gap to best / count of the pair's ``nameaddr_cos`` among
  the query's candidates per source and overall, and among all queries that retrieved the same target.

There is no phone / e-mail / website field in the data; "domain-like name" is available only as a flag.
"""

import numpy as np
from numba import njit, prange

from . import blocking as B
from . import normalize as N

SIM_NAMES = ["nq", "nt", "inter", "jacc", "cont_q", "cont_t", "cos", "wjacc"]
TARGET_FLAGS = [("t_name_missing", N.FLAG_NAME_MISSING), ("t_addr_missing", N.FLAG_ADDR_MISSING),
                ("t_name_nonlatin", N.FLAG_NAME_NONLATIN), ("t_name_domain", N.FLAG_NAME_DOMAIN),
                ("t_name_junk", N.FLAG_NAME_JUNK_PREFIX), ("t_postal", N.FLAG_POSTAL_PRESENT),
                ("t_landmark", N.FLAG_ADDR_LANDMARK)]
QUERY_FLAGS = [("q_postal", N.FLAG_POSTAL_PRESENT), ("q_landmark", N.FLAG_ADDR_LANDMARK)]
FEATURE_NAMES = (["is_s3"] + [f"eq_{k}" for k in N.KEY_COLS] + ["postal_both", "house_both", "digits_both"]
                 + [n for n, _ in TARGET_FLAGS] + [n for n, _ in QUERY_FLAGS]
                 + [f"name_{s}" for s in SIM_NAMES] + [f"addr_{s}" for s in SIM_NAMES] + ["nameaddr_cos"]
                 + ["rank_src", "gap_src", "n_cand_src", "rank_all", "gap_all", "n_cand_all"]
                 + ["t_rank", "t_gap", "t_ncand"])
COL = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)


# ------------------------------------------------------------------------------------------------
# IDF tables (hash table token -> df over the candidate pool, both sources together)
# ------------------------------------------------------------------------------------------------

class IdfTable:
    def __init__(self, keys, vals, idf, n_docs):
        self.keys, self.vals, self.idf, self.n_docs = keys, vals, idf, n_docs
        self.unseen = np.float32(np.log(n_docs + 1.0) + 1.0)

    def save(self, path):
        np.savez(path, keys=self.keys, vals=self.vals, idf=self.idf, n_docs=np.array([self.n_docs]))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["keys"], z["vals"], z["idf"], int(z["n_docs"][0]))


def build_idf(pool_stores, pool_rows, channel):
    toks = [B.gather_csr(*pool_stores[k].csr(channel), np.asarray(pool_rows[k], dtype=np.int64))[1]
            for k in (2, 3)]
    tok = np.concatenate(toks)
    del toks
    n_docs = int(len(pool_rows[2]) + len(pool_rows[3]))
    keys, vals, n_vocab, inv = B._vocab_build(tok)
    del tok
    df = np.bincount(inv, minlength=n_vocab)
    idf = (np.log((n_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    return IdfTable(keys, vals, idf, n_docs)


# ------------------------------------------------------------------------------------------------
# numba kernels
# ------------------------------------------------------------------------------------------------

@njit(inline="always")
def _lookup(keys, vals, k):
    cap = keys.shape[0]
    slot = k & (cap - 1)
    while True:
        kk = keys[slot]
        if kk == 0:
            return -1
        if kk == k:
            return vals[slot]
        slot = (slot + 1) & (cap - 1)


@njit(inline="always")
def _w(keys, vals, idf, unseen, k):
    j = _lookup(keys, vals, k)
    w = idf[j] if j >= 0 else unseen
    return w * w


@njit(cache=True)
def set_sims(ta, tb, keys, vals, idf, unseen):
    """(n_a, n_b, n_inter, Σ_a idf², Σ_b idf², Σ_inter idf²) for two token sets."""
    sa = np.sort(ta)
    sb = np.sort(tb)
    na, nb = sa.shape[0], sb.shape[0]
    wa = 0.0
    wb = 0.0
    wi = 0.0
    for x in sa:
        wa += _w(keys, vals, idf, unseen, x)
    for x in sb:
        wb += _w(keys, vals, idf, unseen, x)
    i = 0
    j = 0
    inter = 0
    while i < na and j < nb:
        if sa[i] == sb[j]:
            wi += _w(keys, vals, idf, unseen, sa[i])
            inter += 1
            i += 1
            j += 1
        elif sa[i] < sb[j]:
            i += 1
        else:
            j += 1
    return na, nb, inter, wa, wb, wi


@njit(inline="always")
def _fill(out, i, c0, na, nb, inter, wa, wb, wi):
    out[i, c0 + 0] = na
    out[i, c0 + 1] = nb
    out[i, c0 + 2] = inter
    u = na + nb - inter
    out[i, c0 + 3] = inter / u if u > 0 else 0.0
    out[i, c0 + 4] = inter / na if na > 0 else 0.0
    out[i, c0 + 5] = inter / nb if nb > 0 else 0.0
    out[i, c0 + 6] = wi / np.sqrt(wa * wb) if wa > 0 and wb > 0 else 0.0
    wu = wa + wb - wi
    out[i, c0 + 7] = wi / wu if wu > 0 else 0.0


@njit(parallel=True, cache=True)
def _sim_kernel(q_rows, src, t_rows,
                q_np, q_nt, q_ap, q_at, s2_np, s2_nt, s2_ap, s2_at, s3_np, s3_nt, s3_ap, s3_at,
                nk, nv, nidf, nun, ak, av, aidf, aun, out, c_name, c_addr, c_na):
    for i in prange(q_rows.shape[0]):
        r = q_rows[i]
        t = t_rows[i]
        qn = q_nt[q_np[r]:q_np[r + 1]]
        qa = q_at[q_ap[r]:q_ap[r + 1]]
        if src[i] == 2:
            tn = s2_nt[s2_np[t]:s2_np[t + 1]]
            ta = s2_at[s2_ap[t]:s2_ap[t + 1]]
        else:
            tn = s3_nt[s3_np[t]:s3_np[t + 1]]
            ta = s3_at[s3_ap[t]:s3_ap[t + 1]]
        na, nb, ni, wa, wb, wi = set_sims(qn, tn, nk, nv, nidf, nun)
        _fill(out, i, c_name, na, nb, ni, wa, wb, wi)
        aa, ab, ai, xa, xb, xi = set_sims(qa, ta, ak, av, aidf, aun)
        _fill(out, i, c_addr, aa, ab, ai, xa, xb, xi)
        den = np.sqrt((wa + xa) * (wb + xb))
        out[i, c_na] = (wi + xi) / den if den > 0 else 0.0


def _group_context(group_keys, score, tie):
    """rank (0 = best), gap to best score, group size — groups defined by ``group_keys`` (tuple of arrays)."""
    order = np.lexsort((tie, -score) + tuple(reversed(group_keys)))
    gk = np.zeros(len(order), dtype=bool)
    for k in group_keys:
        ks = k[order]
        gk[1:] |= ks[1:] != ks[:-1]
    gk[0] = True
    starts = np.flatnonzero(gk)
    sizes = np.diff(np.append(starts, len(order)))
    gid = np.repeat(np.arange(len(starts)), sizes)
    pos = np.arange(len(order)) - starts[gid]
    best = score[order][starts][gid]
    rank = np.empty(len(order), dtype=np.float32)
    gap = np.empty(len(order), dtype=np.float32)
    size = np.empty(len(order), dtype=np.float32)
    rank[order] = pos
    gap[order] = best - score[order]
    size[order] = sizes[gid]
    return rank, gap, size


# ------------------------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------------------------

def pair_features(q_store, q_file_rows, pool_stores, codes, idf_name, idf_addr):
    """Features for pair ``codes`` (sorted; ``ber.blocking`` encoding) of ONE chunk that contains all
    candidates of its queries. ``q_file_rows[q]`` = S1 file row of query position q.
    The three target-context columns are left 0 — fill them with ``target_context`` over the full set."""
    codes = np.asarray(codes, dtype=np.int64)
    q, src, t = B.decode(codes)
    qr = np.asarray(q_file_rows, dtype=np.int64)[q]
    n = len(codes)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    X[:, COL["is_s3"]] = src == 3
    s2 = src == 2
    s3 = ~s2

    def tcol(name):
        out = np.empty(n, dtype=np.asarray(pool_stores[2].col(name)[:1]).dtype)
        out[s2] = np.asarray(pool_stores[2].col(name))[t[s2]]
        out[s3] = np.asarray(pool_stores[3].col(name))[t[s3]]
        return out

    for k in N.KEY_COLS:
        a = np.asarray(q_store.col(k))[qr]
        b = tcol(k)
        X[:, COL[f"eq_{k}"]] = (a == b) & (a != 0)
        if k in ("postal", "house", "addr_digits"):
            X[:, COL[{"postal": "postal_both", "house": "house_both", "addr_digits": "digits_both"}[k]]] = \
                (a != 0) & (b != 0)
    ft = tcol("flags").astype(np.int64)
    fq = np.asarray(q_store.col("flags"))[qr].astype(np.int64)
    for name, bit in TARGET_FLAGS:
        X[:, COL[name]] = (ft & bit) > 0
    for name, bit in QUERY_FLAGS:
        X[:, COL[name]] = (fq & bit) > 0
    qn, qa = q_store.csr("name"), q_store.csr("addr")
    s2n, s2a = pool_stores[2].csr("name"), pool_stores[2].csr("addr")
    s3n, s3a = pool_stores[3].csr("name"), pool_stores[3].csr("addr")
    A = np.asarray
    _sim_kernel(qr, src.astype(np.int64), t,
                A(qn[0]), A(qn[1]), A(qa[0]), A(qa[1]), A(s2n[0]), A(s2n[1]), A(s2a[0]), A(s2a[1]),
                A(s3n[0]), A(s3n[1]), A(s3a[0]), A(s3a[1]),
                idf_name.keys, idf_name.vals, idf_name.idf, idf_name.unseen,
                idf_addr.keys, idf_addr.vals, idf_addr.idf, idf_addr.unseen,
                X, COL["name_nq"], COL["addr_nq"], COL["nameaddr_cos"])
    score = X[:, COL["nameaddr_cos"]].astype(np.float64)
    tie = (src.astype(np.int64) << 32) | t
    r, g, s = _group_context((q, src), score, tie)
    X[:, COL["rank_src"]], X[:, COL["gap_src"]], X[:, COL["n_cand_src"]] = r, g, s
    r, g, s = _group_context((q,), score, tie)
    X[:, COL["rank_all"]], X[:, COL["gap_all"]], X[:, COL["n_cand_all"]] = r, g, s
    return X


def target_context(codes, nameaddr_cos):
    """rank / gap / count of each pair among all pairs sharing its target record (whole candidate set)."""
    q, src, t = B.decode(np.asarray(codes, dtype=np.int64))
    tkey = (src.astype(np.int64) << 32) | t
    return _group_context((tkey,), np.asarray(nameaddr_cos, dtype=np.float64), q)
