"""Streaming, chunked, multi-process featurisation of a source TSV into an on-disk feature store.

For ``<something>_source{1,2,3}.tsv`` this writes, under ``<store_dir>/<name>/``:

    id.npy                 int64   numeric part of entity_id (canonical "S<k>-<int>" verified)
    <key>.npy              int64   blake2b-64 hash of each exact key in normalize.KEY_COLS (0 = empty)
    flags.npy              uint8   normalize.FLAG_* bitmask
    name_indptr.npy        int64   CSR row pointers (n+1) into name_tok
    name_tok.npy           int64   hashed name tokens (unique per record)
    addr_indptr.npy        int64
    addr_tok.npy           int64   hashed address tokens (unique per record)
    meta.json              source path, size/mtime stamp, prefix, rows, NORM_VERSION

Rows are in file order (header skipped, blank lines skipped) — identical indexing to the EDA cache and
to ``assignment/*_fold.npy`` of the validation split. The file is split into line-aligned byte ranges
that worker processes normalise independently; peak memory is one chunk per worker.
"""

import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import normalize as N

CHUNK_LINES = 250_000
FEATURE_VERSION = 1


def _chunk_offsets(path, chunk_lines=CHUNK_LINES):
    """Byte offsets [start, end) of line-aligned chunks covering the data rows (header excluded)."""
    offsets = []
    with open(path, "rb") as f:
        f.readline()
        start = f.tell()
        n = 0
        pos = start
        for line in f:
            pos += len(line)
            n += 1
            if n == chunk_lines:
                offsets.append((start, pos))
                start, n = pos, 0
        if n:
            offsets.append((start, pos))
    return offsets


def _featurise_chunk(task):
    path, start, end, prefix, out_path = task
    ids, flags = [], []
    keys = {c: [] for c in N.KEY_COLS}
    name_len, name_tok, addr_len, addr_tok = [], [], [], []
    h = N.h64
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    for b in data.split(b"\n"):
        if not b.strip():
            continue
        line = b.decode("utf-8").rstrip("\r")
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        eid, name, addr, country = parts
        eid = eid.strip()
        if not eid.startswith(prefix):
            raise ValueError(f"{path}: entity_id {eid!r} does not start with {prefix}")
        num = int(eid[len(prefix):])
        if f"{prefix}{num}" != eid:
            raise ValueError(f"{path}: non-canonical entity_id {eid!r} (leading zeros?)")
        rec = N.normalize_record(name, addr, country)
        ids.append(num)
        for c in N.KEY_COLS:
            keys[c].append(h(rec[c]))
        flags.append(N.record_flags(name, addr, country, rec))
        nt, at = N.blocking_tokens(rec, name.strip())
        name_len.append(len(nt))
        name_tok.extend(h(t) for t in nt)
        addr_len.append(len(at))
        addr_tok.extend(h(t) for t in at)
    np.savez(out_path,
             id=np.array(ids, dtype=np.int64), flags=np.array(flags, dtype=np.uint8),
             name_len=np.array(name_len, dtype=np.int32), name_tok=np.array(name_tok, dtype=np.int64),
             addr_len=np.array(addr_len, dtype=np.int32), addr_tok=np.array(addr_tok, dtype=np.int64),
             **{f"key_{c}": np.array(v, dtype=np.int64) for c, v in keys.items()})
    return len(ids)


def _stamp(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def featurise_file(path, store_dir, name, prefix, workers=4, log=print):
    """Featurise ``path`` into ``store_dir/name``; skipped if an up-to-date store already exists."""
    path, out = Path(path), Path(store_dir) / name
    meta_path = out / "meta.json"
    stamp = _stamp(path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("stamp") == stamp and meta.get("norm_version") == N.NORM_VERSION \
                and meta.get("feature_version") == FEATURE_VERSION:
            log(f"  {name}: up to date ({meta['rows']:,} rows)")
            return meta
    t0 = time.time()
    tmp = out / "_chunks"
    if out.exists():
        shutil.rmtree(out)
    tmp.mkdir(parents=True)
    offsets = _chunk_offsets(path)
    tasks = [(str(path), s, e, prefix, str(tmp / f"c{i:05d}.npz")) for i, (s, e) in enumerate(offsets)]
    with Pool(workers) as pool:
        counts = pool.map(_featurise_chunk, tasks, chunksize=1)
    n = int(sum(counts))

    def open_mm(fname, dtype, length):
        return np.lib.format.open_memmap(out / fname, mode="w+", dtype=dtype, shape=(length,))

    # pass 1: sizes
    n_name = n_addr = 0
    for t in tasks:
        with np.load(t[4]) as z:
            n_name += len(z["name_tok"])
            n_addr += len(z["addr_tok"])
    arrs = {"id": open_mm("id.npy", np.int64, n), "flags": open_mm("flags.npy", np.uint8, n)}
    for c in N.KEY_COLS:
        arrs[c] = open_mm(f"{c}.npy", np.int64, n)
    name_indptr = open_mm("name_indptr.npy", np.int64, n + 1)
    addr_indptr = open_mm("addr_indptr.npy", np.int64, n + 1)
    name_tok = open_mm("name_tok.npy", np.int64, n_name)
    addr_tok = open_mm("addr_tok.npy", np.int64, n_addr)
    name_indptr[0] = addr_indptr[0] = 0
    r = pn = pa = 0
    for t in tasks:
        with np.load(t[4]) as z:
            k = len(z["id"])
            arrs["id"][r:r + k] = z["id"]
            arrs["flags"][r:r + k] = z["flags"]
            for c in N.KEY_COLS:
                arrs[c][r:r + k] = z[f"key_{c}"]
            name_indptr[r + 1:r + k + 1] = pn + np.cumsum(z["name_len"])
            addr_indptr[r + 1:r + k + 1] = pa + np.cumsum(z["addr_len"])
            name_tok[pn:pn + len(z["name_tok"])] = z["name_tok"]
            addr_tok[pa:pa + len(z["addr_tok"])] = z["addr_tok"]
            pn += len(z["name_tok"])
            pa += len(z["addr_tok"])
            r += k
    for a in list(arrs.values()) + [name_indptr, addr_indptr, name_tok, addr_tok]:
        a.flush()
    del arrs, name_indptr, addr_indptr, name_tok, addr_tok
    shutil.rmtree(tmp)
    if len(np.unique(np.load(out / "id.npy"))) != n:
        raise ValueError(f"{path}: duplicate entity IDs")
    meta = {"source": str(path), "stamp": stamp, "prefix": prefix, "rows": n,
            "name_postings": n_name, "addr_postings": n_addr, "norm_version": N.NORM_VERSION,
            "feature_version": FEATURE_VERSION, "seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=1))
    log(f"  {name}: {n:,} rows, {n_name:,} name + {n_addr:,} address postings ({meta['seconds']}s)")
    return meta


class FeatureStore:
    """Read-only, memory-mapped view of one featurised source file."""

    def __init__(self, store_dir, name):
        self.dir = Path(store_dir) / name
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.prefix = self.meta["prefix"]
        self.n = self.meta["rows"]

    def col(self, name):
        return np.load(self.dir / f"{name}.npy", mmap_mode="r")

    def csr(self, channel):
        return self.col(f"{channel}_indptr"), self.col(f"{channel}_tok")

    def entity_ids(self, rows=None):
        ids = self.col("id")
        ids = ids if rows is None else ids[rows]
        return [f"{self.prefix}{int(x)}" for x in ids]


def featurise_dir(test_dir, store_dir, workers=4, stem="test", log=print):
    """Featurise ``<test_dir>/<stem>_source{1,2,3}.tsv`` (competition layout)."""
    out = {}
    for k in (1, 2, 3):
        out[f"s{k}"] = featurise_file(Path(test_dir) / f"{stem}_source{k}.tsv", store_dir, f"{stem}_s{k}",
                                      f"S{k}-", workers=workers, log=log)
    return out
