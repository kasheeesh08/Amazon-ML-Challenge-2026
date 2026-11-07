"""Orchestration helpers of the final pipeline (consolidated from the project's experiment drivers).

Everything here is logic that the validated run used, moved out of the experiment scripts so the package is
self-contained:

* ``candidate_union``      — the Step 2 candidate set for a batch of S1 queries: every configured channel is
  generated for S2 and S3, merged into a running union and freed (``experiments/blocking_experiment.lean_union``).
* ``iter_query_chunks``    — query-aligned chunks of a sorted (memory-mapped) pair-code array.
* ``target_context_partitioned`` — ``match_features.target_context`` computed per target-row partition
  (identical result, bounded memory).
* ``load_truth``           — training ground truth as pair codes (reads ``train_ground_truth.tsv`` directly).
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import blocking as B
from . import match_features as MF

MIN_FREE_BYTES = 1_000_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(conv(obj), indent=1), encoding="utf-8")
    tmp.replace(path)


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def memory():
    """Process/system memory snapshot (psutil is optional)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return {"peak_working_set_gb": round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2),
                "private_gb": round(getattr(mi, "private", mi.rss) / 1e9, 2),
                "system_free_gb": round(psutil.virtual_memory().available / 1e9, 2)}
    except Exception:
        return {}


def guard(stage):
    """Stop cleanly (instead of being killed) if free RAM drops below 1 GB (psutil optional)."""
    try:
        import psutil
        fb = psutil.virtual_memory().available
    except Exception:
        return
    if fb < MIN_FREE_BYTES:
        raise MemoryError(f"free RAM {fb / 1e9:.2f} GB < {MIN_FREE_BYTES / 1e9:.1f} GB before {stage}")


class IdIndex:
    """Vectorised lookup: numeric entity ID -> row index in its source file (-1 if absent)."""

    def __init__(self, ids):
        ids = np.asarray(ids)
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]

    def rows(self, q):
        q = np.asarray(q)
        pos = np.minimum(np.searchsorted(self.sorted, q), len(self.sorted) - 1)
        return np.where(self.sorted[pos] == q, self.order[pos], -1).astype(np.int64)


def candidate_union(stores, config, q_rows, pool_rows):
    """Union of all configured blocking channels for S1 rows ``q_rows`` (codes use positions in q_rows).

    ``stores``: {1: S1 FeatureStore, 2: S2, 3: S3}; ``pool_rows``: {2: rows, 3: rows}."""
    union = np.empty(0, dtype=np.int64)
    info = {}
    jobs = [("exact:" + "+".join(ex["key"]), "exact", ex) for ex in config.get("exact", [])] + \
           [(f"retrieval:{rt['channel']}@{rt['k']}", "retrieval", rt) for rt in config.get("retrieval", [])]
    for name, kind, spec in jobs:
        guard(name)
        t0 = time.time()
        parts = []
        for src in (2, 3):
            if kind == "exact":
                qk = B.key_array(stores[1], tuple(spec["key"]), q_rows)
                pk = B.key_array(stores[src], tuple(spec["key"]), pool_rows[src])
                parts.append(B.exact_channel(qk, pk, pool_rows[src], src, spec.get("cap")))
                del qk, pk
            else:
                c, _, _ = B.retrieval_channel(stores[1], q_rows, stores[src], pool_rows[src], src,
                                              spec["channel"], spec["k"], spec["max_postings"])
                parts.append(c)
        codes = np.unique(np.concatenate(parts))
        del parts
        info[name] = {"n_pairs": int(len(codes)), "seconds": round(time.time() - t0, 1)}
        union = np.union1d(union, codes)
        del codes
    return union, info


def iter_query_chunks(codes, chunk=2_000_000):
    """Query-aligned [a, b) chunks of a sorted (memory-mapped) code array, found by reading small windows."""
    n = len(codes)
    a = 0
    while a < n:
        b = min(n, a + chunk)
        if b < n:
            qb = int(codes[b]) >> B.Q_SHIFT
            w = 1024
            while True:
                lo = max(a, b - w)
                seg = np.asarray(codes[lo:b]) >> B.Q_SHIFT
                if lo == a or seg[0] != qb:
                    break
                w *= 2
            b = lo + int(np.searchsorted(seg, qb, "left"))
            if b == a:
                w = 1024
                while True:
                    hi = min(n, a + chunk + w)
                    seg = np.asarray(codes[a:hi]) >> B.Q_SHIFT
                    if hi == n or seg[-1] != qb:
                        break
                    w *= 2
                b = a + int(np.searchsorted(seg, qb, "right"))
        yield a, b
        a = b


def target_context_partitioned(codes_path, cos_path, out_prefix, nrows2, nrows3, tmp_index, parts=8):
    """t_rank / t_gap / t_ncand for every pair — identical to MF.target_context over the whole set."""
    codes = np.load(codes_path, mmap_mode="r")
    cos = np.load(cos_path, mmap_mode="r")
    n = len(codes)
    P = parts

    def part_of(c):
        _, s, t = B.decode(c)
        return (s - 2) * P + np.minimum((t * P) // np.where(s == 2, nrows2, nrows3), P - 1)
    step = 20_000_000
    counts = np.zeros(2 * P, dtype=np.int64)
    for a in range(0, n, step):
        counts += np.bincount(part_of(np.asarray(codes[a:a + step])), minlength=2 * P)
    starts = np.concatenate([[0], np.cumsum(counts)])
    idx = np.lib.format.open_memmap(tmp_index, mode="w+", dtype=np.int64, shape=(n,))
    fill = starts[:-1].copy()
    for a in range(0, n, step):
        pid = part_of(np.asarray(codes[a:a + step]))
        order = np.argsort(pid, kind="stable")
        bnd = np.searchsorted(pid[order], np.arange(2 * P + 1))
        for k in range(2 * P):
            seg = order[bnd[k]:bnd[k + 1]] + a
            idx[fill[k]:fill[k] + len(seg)] = seg
            fill[k] += len(seg)
    outs = {nm: np.lib.format.open_memmap(f"{out_prefix}_{nm}.npy", mode="w+", dtype=np.float32, shape=(n,))
            for nm in ("t_rank", "t_gap", "t_ncand")}
    for k in range(2 * P):
        sel = np.sort(np.asarray(idx[starts[k]:starts[k + 1]]))
        if len(sel):
            r, g, s = MF.target_context(np.asarray(codes[sel]), np.asarray(cos[sel]))
            outs["t_rank"][sel], outs["t_gap"][sel], outs["t_ncand"][sel] = r, g, s
    for o in outs.values():
        o.flush()
    del idx, outs
    Path(tmp_index).unlink()


def load_truth(gt_path, stores):
    """Training ground truth as sorted unique pair codes (q = S1 file row, src, t = S2/S3 file row)."""
    idx = {k: IdIndex(np.asarray(stores[k].col("id"))) for k in (1, 2, 3)}
    s1, src, tnum = [], [], []
    with open(gt_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, rest = line.rstrip("\n").partition("\t")
            if not rest.strip():
                continue
            for x in rest.split(","):
                s1.append(int(a.strip()[3:]))
                src.append(int(x[1]))
                tnum.append(int(x[3:]))
    s1, src, tnum = np.array(s1, np.int64), np.array(src, np.int64), np.array(tnum, np.int64)
    q = idx[1].rows(s1)
    t = np.full(len(tnum), -1, dtype=np.int64)
    for k in (2, 3):
        m = src == k
        t[m] = idx[k].rows(tnum[m])
    ok = (q >= 0) & (t >= 0)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} ground-truth IDs not found in the training sources")
    return np.unique(B.encode(q, src, t))
