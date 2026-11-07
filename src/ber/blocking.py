"""Candidate generation (blocking). Independent of — and not to be confused with — the matcher.

Channels
--------
* **Exact-key channels** — equality join on a normalised key (or a conjunction of keys), per source.
  Blocks larger than ``cap`` pool records are dropped (generic keys such as "meridian" would
  otherwise explode the candidate set).
* **Retrieval channels** — TF-IDF cosine top-K over hashed tokens with an inverted index:
  ``name`` (name words + character trigrams), ``addr`` (address tokens, digit groups, house number),
  ``nameaddr`` (both). For every query the rarest tokens are used first until ``max_postings`` postings
  have been scanned (the rarest token is always used), so the cost per query is bounded. Retrieval is
  run separately for Source 2 and Source 3 (``k`` candidates each — sources stay distinguishable and
  one source cannot crowd out the other) and within the query's country (EDA: 0 of 7.64M true pairs
  cross countries). A query whose country is empty or absent from the pool falls back to the whole
  pool of that source.

Candidate pairs are encoded as int64 ``(q << 33) | ((src - 2) << 32) | t`` where ``q`` is the query's
position in the query row list, ``src`` is 2 or 3 and ``t`` is the record's row in its source file.
Nothing here needs labels; evaluation lives in ``ber.blocking_eval``.
"""

import numba
import numpy as np
from numba import njit, prange

Q_SHIFT = 33
SRC_SHIFT = 32
T_MASK = (1 << 32) - 1


def encode(q, src, t):
    q = np.asarray(q, dtype=np.int64)
    return (q << Q_SHIFT) | ((np.asarray(src, dtype=np.int64) - 2) << SRC_SHIFT) | np.asarray(t, dtype=np.int64)


def decode(codes):
    codes = np.asarray(codes, dtype=np.int64)
    return codes >> Q_SHIFT, ((codes >> SRC_SHIFT) & 1) + 2, codes & T_MASK


# ------------------------------------------------------------------------------------------------
# CSR helpers (numba, no large temporaries)
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def _gather_into(indptr, tok, rows, new_ptr, out):
    for i in range(rows.shape[0]):
        s = indptr[rows[i]]
        e = indptr[rows[i] + 1]
        o = new_ptr[i]
        for j in range(e - s):
            out[o + j] = tok[s + j]


def gather_csr(indptr, tok, rows):
    """Sub-CSR of ``rows`` (in that order) from a (possibly memory-mapped) CSR."""
    indptr = np.asarray(indptr)
    rows = np.asarray(rows, dtype=np.int64)
    lens = indptr[rows + 1] - indptr[rows]
    new_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=new_ptr[1:])
    out = np.empty(int(new_ptr[-1]), dtype=np.int64)
    _gather_into(indptr, np.asarray(tok), rows, new_ptr, out)
    return new_ptr, out


@njit(cache=True)
def _merge_into(pa, ta, pb, tb, new_ptr, out):
    for i in range(pa.shape[0] - 1):
        o = new_ptr[i]
        for j in range(pa[i], pa[i + 1]):
            out[o] = ta[j]
            o += 1
        for j in range(pb[i], pb[i + 1]):
            out[o] = tb[j]
            o += 1


@njit(cache=True)
def _gather_merged_into(pa, ta, pb, tb, rows, new_ptr, out):
    for i in range(rows.shape[0]):
        r = rows[i]
        o = new_ptr[i]
        for j in range(pa[r], pa[r + 1]):
            out[o] = ta[j]
            o += 1
        for j in range(pb[r], pb[r + 1]):
            out[o] = tb[j]
            o += 1


def gather_merged_csr(a, b, rows):
    """Row-wise concatenation of two CSRs, gathered for ``rows`` directly into one array."""
    pa, ta = np.asarray(a[0]), np.asarray(a[1])
    pb, tb = np.asarray(b[0]), np.asarray(b[1])
    rows = np.asarray(rows, dtype=np.int64)
    lens = (pa[rows + 1] - pa[rows]) + (pb[rows + 1] - pb[rows])
    new_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=new_ptr[1:])
    out = np.empty(int(new_ptr[-1]), dtype=np.int64)
    _gather_merged_into(pa, ta, pb, tb, rows, new_ptr, out)
    return new_ptr, out


def merge_csr(a, b):
    """Row-wise concatenation of two CSRs over the same rows (token namespaces never collide)."""
    pa, ta = a
    pb, tb = b
    new_ptr = pa + pb
    out = np.empty(int(new_ptr[-1]), dtype=np.int64)
    _merge_into(pa, ta, pb, tb, new_ptr, out)
    return new_ptr, out


# ------------------------------------------------------------------------------------------------
# Inverted index + top-K retrieval
# ------------------------------------------------------------------------------------------------

@njit(cache=True)
def _vocab_build(tok):
    """Assign dense ids (first-seen order) to non-zero int64 token hashes via linear probing.

    Token hashes are blake2b outputs (uniform low bits, never 0), so ``hash & (cap-1)`` is a good slot.
    Returns (keys, vals, n_vocab, inv) where inv[i] is the id of tok[i]."""
    cap = 1 << 20
    keys = np.zeros(cap, dtype=np.int64)
    vals = np.empty(cap, dtype=np.int32)
    inv = np.empty(tok.shape[0], dtype=np.int32)
    n = 0
    for i in range(tok.shape[0]):
        k = tok[i]
        slot = k & (cap - 1)
        while True:
            kk = keys[slot]
            if kk == 0:
                keys[slot] = k
                vals[slot] = n
                inv[i] = n
                n += 1
                break
            if kk == k:
                inv[i] = vals[slot]
                break
            slot = (slot + 1) & (cap - 1)
        if 2 * n > cap:                                   # keep load factor <= 0.5
            new_cap = cap * 2
            nk = np.zeros(new_cap, dtype=np.int64)
            nv = np.empty(new_cap, dtype=np.int32)
            for s in range(cap):
                if keys[s] != 0:
                    t = keys[s] & (new_cap - 1)
                    while nk[t] != 0:
                        t = (t + 1) & (new_cap - 1)
                    nk[t] = keys[s]
                    nv[t] = vals[s]
            keys, vals, cap = nk, nv, new_cap
    return keys, vals, n, inv


@njit(cache=True)
def _vocab_lookup(keys, vals, q_tok):
    cap = keys.shape[0]
    out = np.full(q_tok.shape[0], -1, dtype=np.int32)
    for i in range(q_tok.shape[0]):
        k = q_tok[i]
        if k == 0:
            continue
        slot = k & (cap - 1)
        while True:
            kk = keys[slot]
            if kk == 0:
                break
            if kk == k:
                out[i] = vals[slot]
                break
            slot = (slot + 1) & (cap - 1)
    return out


@njit(cache=True)
def _build_inverted(ptr, inv, df, n_vocab):
    inv_ptr = np.zeros(n_vocab + 1, dtype=np.int64)
    for t in range(n_vocab):
        inv_ptr[t + 1] = inv_ptr[t] + df[t]
    fill = inv_ptr[:-1].copy()
    docs = np.empty(inv_ptr[-1], dtype=np.int32)
    for d in range(ptr.shape[0] - 1):
        for p in range(ptr[d], ptr[d + 1]):
            t = inv[p]
            docs[fill[t]] = d
            fill[t] += 1
    return inv_ptr, docs


@njit(cache=True)
def _doc_norms(ptr, inv, w):
    n = ptr.shape[0] - 1
    out = np.empty(n, dtype=np.float32)
    for d in range(n):
        s = 0.0
        for p in range(ptr[d], ptr[d + 1]):
            s += w[inv[p]]
        out[d] = np.sqrt(s) if s > 0 else 1.0
    return out


@njit(cache=True, inline="always")
def _worse(sa, da, sb, db):
    """True if (score sa, doc da) ranks below (sb, db): lower score, or equal score and larger doc."""
    return sa < sb or (sa == sb and da > db)


@njit(parallel=True, cache=True)
def _topk(q_ptr, q_inv, df, w, inv_ptr, inv_docs, doc_norm, n_docs, k, max_postings, chunk):
    nq = q_ptr.shape[0] - 1
    out_doc = np.full((nq, k), -1, dtype=np.int32)
    out_sc = np.zeros((nq, k), dtype=np.float32)
    n_chunks = (nq + chunk - 1) // chunk
    for c in prange(n_chunks):
        acc = np.zeros(n_docs, dtype=np.float32)
        touched = np.empty(n_docs, dtype=np.int32)
        heap_sc = np.empty(k, dtype=np.float32)
        heap_doc = np.empty(k, dtype=np.int32)
        for qi in range(c * chunk, min(nq, (c + 1) * chunk)):
            s = q_ptr[qi]
            e = q_ptr[qi + 1]
            m = e - s
            if m == 0:
                continue
            dfs = np.empty(m, dtype=np.int64)
            for j in range(m):
                t = q_inv[s + j]
                dfs[j] = df[t] if t >= 0 else np.int64(1) << 62
            order = np.argsort(dfs, kind="mergesort")
            nt = 0
            used = 0
            for jj in range(m):
                j = order[jj]
                t = q_inv[s + j]
                if t < 0:
                    break
                d = df[t]
                if used > 0 and used + d > max_postings:
                    break
                used += d
                wt = w[t]
                for p in range(inv_ptr[t], inv_ptr[t + 1]):
                    doc = inv_docs[p]
                    if acc[doc] == 0.0:
                        touched[nt] = doc
                        nt += 1
                    acc[doc] += wt
            if nt == 0:
                continue
            # bounded min-heap of the k best (score desc, doc asc) — O(nt log k), deterministic
            size = 0
            for j in range(nt):
                doc = touched[j]
                s_ = acc[doc] / doc_norm[doc]
                acc[doc] = 0.0
                if size < k:
                    heap_sc[size] = s_
                    heap_doc[size] = doc
                    i = size
                    size += 1
                    while i > 0:                                   # sift up
                        par = (i - 1) // 2
                        if _worse(heap_sc[i], heap_doc[i], heap_sc[par], heap_doc[par]):
                            heap_sc[i], heap_sc[par] = heap_sc[par], heap_sc[i]
                            heap_doc[i], heap_doc[par] = heap_doc[par], heap_doc[i]
                            i = par
                        else:
                            break
                elif _worse(heap_sc[0], heap_doc[0], s_, doc):
                    heap_sc[0] = s_
                    heap_doc[0] = doc
                    i = 0
                    while True:                                    # sift down
                        lft = 2 * i + 1
                        rgt = lft + 1
                        m_ = i
                        if lft < size and _worse(heap_sc[lft], heap_doc[lft], heap_sc[m_], heap_doc[m_]):
                            m_ = lft
                        if rgt < size and _worse(heap_sc[rgt], heap_doc[rgt], heap_sc[m_], heap_doc[m_]):
                            m_ = rgt
                        if m_ == i:
                            break
                        heap_sc[i], heap_sc[m_] = heap_sc[m_], heap_sc[i]
                        heap_doc[i], heap_doc[m_] = heap_doc[m_], heap_doc[i]
                        i = m_
            # emit best-first: selection over the (small) heap
            for r in range(size):
                best = -1
                for j in range(size):
                    if heap_doc[j] >= 0 and (best < 0 or _worse(heap_sc[best], heap_doc[best],
                                                                  heap_sc[j], heap_doc[j])):
                        best = j
                out_doc[qi, r] = heap_doc[best]
                out_sc[qi, r] = heap_sc[best]
                heap_doc[best] = -1
    return out_doc, out_sc


def retrieve_topk(q_csr, pool_csr, k, max_postings, chunk=None):
    """Top-``k`` pool documents per query by TF-IDF cosine (binary tf, smooth idf from the pool).

    Returns ``(docs, scores)`` of shape (n_queries, k); docs are local pool indices, -1 = none.
    """
    p_ptr, p_tok = pool_csr
    q_ptr, q_tok = q_csr
    n_docs = len(p_ptr) - 1
    nq = len(q_ptr) - 1
    if n_docs == 0 or nq == 0:
        return np.full((nq, k), -1, dtype=np.int32), np.zeros((nq, k), dtype=np.float32)
    # token hash -> dense id with an open-addressing table: O(postings), no sort, memory ~ vocabulary
    keys, vals, n_vocab, inv = _vocab_build(p_tok)
    df = np.bincount(inv, minlength=n_vocab).astype(np.int64)
    idf = (np.log((n_docs + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    w = (idf * idf).astype(np.float32)
    inv_ptr, inv_docs = _build_inverted(p_ptr, inv, df, n_vocab)
    doc_norm = _doc_norms(p_ptr, inv, w)
    del inv
    q_inv = _vocab_lookup(keys, vals, q_tok)
    del keys, vals
    if chunk is None:
        # each chunk allocates two n_docs-sized scratch arrays: use few, large chunks
        chunk = max(64, -(-nq // (numba.get_num_threads() * 4)))
    return _topk(q_ptr, q_inv, df, w, inv_ptr, inv_docs, doc_norm, n_docs, int(k),
                 int(max_postings), int(chunk))


# ------------------------------------------------------------------------------------------------
# Channels over feature stores
# ------------------------------------------------------------------------------------------------

def key_array(store, spec, rows):
    """Exact-key hash per row; a tuple spec is a conjunction (0 if any component is empty)."""
    if isinstance(spec, str):
        spec = (spec,)
    arrs = [np.asarray(store.col(c))[rows] for c in spec]
    if len(arrs) == 1:
        return arrs[0].astype(np.int64)
    out = np.zeros(len(rows), dtype=np.uint64)
    valid = np.ones(len(rows), dtype=bool)
    m1, sh = np.uint64(0x9E3779B97F4A7C15), np.uint64(29)
    with np.errstate(over="ignore"):
        for a in arrs:
            valid &= a != 0
            out = (out ^ a.astype(np.int64).view(np.uint64)) * m1
            out ^= out >> sh
    res = out.view(np.int64).copy()
    res[res == 0] = 1
    res[~valid] = 0
    return res


def exact_block_sizes(q_keys, pool_keys):
    """For each query: (lo, hi) into the key-sorted pool, and the sorted-pool -> pool index map."""
    pidx = np.flatnonzero(pool_keys != 0)
    order = np.argsort(pool_keys[pidx], kind="stable")
    pidx = pidx[order]
    pk = pool_keys[pidx]
    lo = np.searchsorted(pk, q_keys, "left")
    hi = np.searchsorted(pk, q_keys, "right")
    cnt = hi - lo
    cnt[q_keys == 0] = 0
    return lo, cnt, pidx


def exact_channel(q_keys, pool_keys, pool_rows, src, cap):
    """Candidate codes of an exact-key channel for one source; blocks with > ``cap`` records dropped."""
    lo, cnt, pidx = exact_block_sizes(q_keys, pool_keys)
    if cap is not None:
        cnt = np.where(cnt > cap, 0, cnt)
    total = int(cnt.sum())
    q = np.repeat(np.arange(len(q_keys), dtype=np.int64), cnt)
    start = np.cumsum(cnt) - cnt
    pos = np.arange(total, dtype=np.int64) - np.repeat(start, cnt) + np.repeat(lo, cnt)
    t = np.asarray(pool_rows, dtype=np.int64)[pidx[pos]]
    return encode(q, np.full(total, src), t)


def _channel_csr(store, channel, rows):
    if channel == "nameaddr":
        return gather_merged_csr(store.csr("name"), store.csr("addr"), rows)
    return gather_csr(*store.csr(channel), rows)


def retrieval_channel(q_store, q_rows, p_store, p_rows, src, channel, k, max_postings, log=None):
    """Top-``k`` retrieval for one source, partitioned by country.

    Returns ``(codes, ranks, scores)`` for every non-empty result (rank 0 = best)."""
    q_rows = np.asarray(q_rows, dtype=np.int64)
    p_rows = np.asarray(p_rows, dtype=np.int64)
    qc = np.asarray(q_store.col("country"))[q_rows]
    pc = np.asarray(p_store.col("country"))[p_rows]
    pool_countries = set(np.unique(pc).tolist())
    groups = []
    fallback = []
    for c in np.unique(qc):
        qsel = np.flatnonzero(qc == c)
        if c == 0 or int(c) not in pool_countries:
            fallback.append(qsel)
        else:
            groups.append((qsel, np.flatnonzero(pc == c)))
    if fallback:
        groups.append((np.concatenate(fallback), np.arange(len(p_rows))))
    codes, ranks, scores = [], [], []
    for qsel, psel in groups:
        q_csr = _channel_csr(q_store, channel, q_rows[qsel])
        p_csr = _channel_csr(p_store, channel, p_rows[psel])
        docs, sc = retrieve_topk(q_csr, p_csr, k, max_postings)
        del p_csr
        qi, rk = np.nonzero(docs >= 0)
        codes.append(encode(qsel[qi], np.full(len(qi), src), p_rows[psel[docs[qi, rk]]]))
        ranks.append(rk.astype(np.int16))
        scores.append(sc[qi, rk])
        if log:
            log(f"      {channel} S{src} country-group: {len(qsel):,} queries x {len(psel):,} docs")
    if not codes:
        return np.empty(0, np.int64), np.empty(0, np.int16), np.empty(0, np.float32)
    return np.concatenate(codes), np.concatenate(ranks), np.concatenate(scores)


# ------------------------------------------------------------------------------------------------
# Configured candidate generation
# ------------------------------------------------------------------------------------------------

def generate_candidates(config, q_store, q_rows, pool_stores, pool_rows, log=print):
    """Union of all channels in ``config`` -> sorted unique candidate codes.

    ``config``: {"exact": [{"key": [...], "cap": int}], "retrieval": [{"channel": str, "k": int,
    "max_postings": int}]}; ``pool_stores``/``pool_rows``: {2: ..., 3: ...}.
    Also returns per-channel code arrays (for provenance / diagnostics).
    """
    per_channel = {}
    for src in (2, 3):
        ps, pr = pool_stores[src], np.asarray(pool_rows[src], dtype=np.int64)
        for ex in config.get("exact", []):
            name = "exact:" + "+".join(ex["key"])
            qk = key_array(q_store, tuple(ex["key"]), q_rows)
            pk = key_array(ps, tuple(ex["key"]), pr)
            per_channel.setdefault(name, []).append(exact_channel(qk, pk, pr, src, ex.get("cap")))
        for rt in config.get("retrieval", []):
            name = f"retrieval:{rt['channel']}@{rt['k']}"
            codes, _, _ = retrieval_channel(q_store, q_rows, ps, pr, src, rt["channel"], rt["k"],
                                            rt["max_postings"])
            per_channel.setdefault(name, []).append(codes)
            log(f"    {name} S{src}: {len(codes):,} pairs")
    per_channel = {k: np.unique(np.concatenate(v)) for k, v in per_channel.items()}
    union = np.unique(np.concatenate(list(per_channel.values()))) if per_channel else np.empty(0, np.int64)
    return union, per_channel


def write_candidate_pairs(path, codes, q_ids, pool_ids):
    """Stream candidate codes (sorted) to ``candidate_pairs.tsv``.

    ``q_ids``: list of S1 entity IDs by query position (every query gets a row, possibly empty).
    ``pool_ids``: {2: numeric id array by S2 file row, 3: ... by S3 file row}.
    """
    codes = np.unique(np.asarray(codes, dtype=np.int64))
    q, src, t = decode(codes)
    bounds = np.searchsorted(q, np.arange(len(q_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, s1 in enumerate(q_ids):
            a, b = bounds[i], bounds[i + 1]
            ids = sorted(f"S{int(s)}-{int(pool_ids[int(s)][int(r)])}" for s, r in zip(src[a:b], t[a:b]))
            f.write(f"{s1}\t{','.join(ids)}\n")
