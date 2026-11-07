"""Leakage-safe 80/20 train/validation split at the Source-1 (S1) entity level.

Construction (deterministic, uses only training data — never the test files):

1. **S1 entities** are split 80 % fit / 20 % validation, stratified by
   ``country × min(#true matches, 8) × (has S2 match, has S3 match)``. Within each stratum entities are
   ordered by ``blake2b("<seed>:<entity_id>")`` and the first ``quota`` go to validation. Quotas use
   exact integer largest-remainder rounding so the total is exactly ``round(N · 1/5)`` and every stratum
   gets ⌊n/5⌋ or ⌈n/5⌉. The ordering depends only on entity IDs and the seed — not on file order or the
   numpy version.
2. **True targets** (S2/S3 records listed in the ground truth) inherit the fold of their S1 entity. The
   EDA showed — and this module re-checks — that no S2/S3 record is the target of more than one S1, so
   this assignment is unambiguous and no S1's matches can be split across folds.
3. **Distractors** (S2/S3 records that are nobody's match, ~26 % of each source) are split 20 % / 80 %
   the same way, stratified by ``source × country``. The validation pool therefore has the same
   target : distractor ratio as the full training data, and the fit and validation pools are disjoint.

Outputs (``out_dir``, default ``experiments/validation_split/``):

* ``val_as_test_dir/test_source{1,2,3}.tsv`` — the validation set laid out exactly like
  ``dataset/test/`` (byte-identical rows copied from the train files), so the official validator and
  any future pipeline can be pointed at it unchanged. **These are NOT test data** — the file names only
  mimic the test layout.
* ``val_ground_truth.tsv`` — the ground-truth rows of the validation S1 entities (byte-identical).
* ``assignment/{s1,s2,s3}_fold.npy`` — int8 per row of ``train_source{1,2,3}.tsv`` (file order):
  0 = fit, 1 = validation. With ``iter_fold_lines`` this reconstructs the fit fold without copying it.
* ``manifest.json`` — seed, rule, input file stamps, counts, SHA-256 of every output.
* ``VALIDATION_SPLIT.md`` — human-readable description and statistics; ``verification.json``.

Memory: only numpy arrays from the EDA hash cache (``experiments/_eda_cache``) are loaded; the TSVs
are streamed line by line.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ids_tsv import GROUND_TRUTH_HEADER, read_id_list_tsv, read_source_ids

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "dataset" / "train"
DEFAULT_CACHE_DIR = ROOT / "experiments" / "_eda_cache"
DEFAULT_OUT_DIR = ROOT / "experiments" / "validation_split"
VALIDATOR = ROOT / "utils" / "validate_submission.py"

SEED = 20260925
VAL_NUM, VAL_DEN = 1, 5          # validation fraction = 1/5 = 20 %, exact integer arithmetic
COUNT_CAP = 8                    # match-count strata 0,1,...,7,8+
SOURCE_FILES = {"s1": "train_source1.tsv", "s2": "train_source2.tsv", "s3": "train_source3.tsv"}
GT_FILE = "train_ground_truth.tsv"
CACHE_KEYS = {"s1": "train_s1", "s2": "train_s2", "s3": "train_s3", "gt": "gt"}
VAL_DIR_NAME = "val_as_test_dir"
VAL_GT_NAME = "val_ground_truth.tsv"


# ------------------------------------------------------------------------------------------------
# Deterministic, stratified selection
# ------------------------------------------------------------------------------------------------

def seeded_hash(entity_ids, seed=SEED):
    """uint64 ``blake2b("<seed>:<entity_id>")`` for each string ID (order-independent tie-breaker)."""
    pre = f"{seed}:".encode()
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(pre + e.encode(), digest_size=8).digest(), "little")
         for e in entity_ids),
        dtype=np.uint64, count=len(entity_ids))


def stratified_select(strata, keys, num=VAL_NUM, den=VAL_DEN):
    """Boolean mask selecting exactly ``round(n·num/den)`` items, proportionally within each stratum.

    Items are ranked inside their stratum by ``keys`` (ascending); the first ``quota`` are selected.
    Quotas: ⌊n_s·num/den⌋ plus one extra for the strata with the largest remainders (ties -> lower
    stratum code), so the total is exact and every stratum is within one item of proportional.
    """
    strata = np.asarray(strata, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.uint64)
    n = len(strata)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.lexsort((keys, strata))
    uniq, start, counts = np.unique(strata[order], return_index=True, return_counts=True)
    counts = counts.astype(np.int64)
    base = counts * num // den
    rem = counts * num % den
    target = (n * num * 2 + den) // (2 * den)          # round half up, integer
    extra = int(target - base.sum())
    if extra > 0:
        pri = np.lexsort((uniq, -rem))                  # largest remainder first, then stratum code
        base[pri[:extra]] += 1
    for st, q in zip(start, base):
        mask[order[st:st + q]] = True
    return mask


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

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


def _stamp(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache(cache_dir, key, col):
    return np.load(Path(cache_dir) / f"{key}__{col}.npy")


def _check_cache(cache_dir, data_dir):
    """The EDA cache must have been built from exactly these train files (size + mtime)."""
    for k, fname in list(SOURCE_FILES.items()) + [("gt", GT_FILE)]:
        stamp_path = Path(cache_dir) / f"{CACHE_KEYS[k]}__stamp.json"
        if not stamp_path.exists():
            raise FileNotFoundError(f"missing EDA cache stamp {stamp_path}; run experiments/eda.py first")
        cached = json.loads(stamp_path.read_text())
        cur = _stamp(Path(data_dir) / fname)
        if cached["size"] != cur["size"] or cached["mtime"] != cur["mtime"]:
            raise RuntimeError(f"EDA cache is stale for {fname}: cache {cached} vs file {cur}")


def _iter_data_lines(path):
    """Yield (row_index, raw_bytes_line) for data rows, header skipped, blank lines skipped
    (identical row indexing to the EDA cache)."""
    with open(path, "rb") as f:
        f.readline()
        row = 0
        for b in f:
            if not b.strip():
                continue
            yield row, b
            row += 1


def iter_fold_lines(source, fold, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    """Stream the decoded lines of ``train_source{1,2,3}.tsv`` (``source`` in s1/s2/s3) belonging to
    ``fold`` (0 = fit, 1 = validation), using the saved assignment arrays."""
    assign = np.load(Path(out_dir) / "assignment" / f"{source}_fold.npy", mmap_mode="r")
    for row, b in _iter_data_lines(Path(data_dir) / SOURCE_FILES[source]):
        if assign[row] == fold:
            yield b.decode("utf-8").rstrip("\r\n")


def _copy_selected(src, dst, fold_arr, expected_ids, prefix, stats_key, stats):
    """Stream ``src``; write rows whose fold == 1 to ``dst`` byte-for-byte; verify every row's ID
    against the cache; count countries per (fold, kind)."""
    pre = prefix.encode()
    n = 0
    cnt = Counter()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        header = fi.readline()
        fo.write(header if header.endswith(b"\n") else header + b"\n")
        for row, b in _iter_data_lines(src):
            eid = b.split(b"\t", 1)[0].strip()
            if not eid.startswith(pre) or int(eid[len(pre):]) != int(expected_ids[row]):
                raise RuntimeError(f"{src}: row {row} ID {eid!r} does not match the EDA cache")
            fold = int(fold_arr[row])
            country = b.rstrip(b"\r\n").rsplit(b"\t", 1)[-1].strip().decode("utf-8")
            cnt[(fold, country)] += 1
            if fold == 1:
                fo.write(b if b.endswith(b"\n") else b + b"\n")
            n += 1
    if n != len(expected_ids):
        raise RuntimeError(f"{src}: {n} rows streamed but cache has {len(expected_ids)}")
    stats[stats_key] = {f"{'val' if f else 'fit'}|{c}": v for (f, c), v in sorted(cnt.items())}


# ------------------------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------------------------

def build(data_dir=DEFAULT_DATA_DIR, cache_dir=DEFAULT_CACHE_DIR, out_dir=DEFAULT_OUT_DIR, seed=SEED,
          log=print):
    t0 = time.time()
    data_dir, cache_dir, out_dir = Path(data_dir), Path(cache_dir), Path(out_dir)
    _check_cache(cache_dir, data_dir)
    (out_dir / "assignment").mkdir(parents=True, exist_ok=True)
    (out_dir / VAL_DIR_NAME).mkdir(parents=True, exist_ok=True)

    ids = {k: _load_cache(cache_dir, CACHE_KEYS[k], "id") for k in SOURCE_FILES}
    country = {k: _load_cache(cache_dir, CACHE_KEYS[k], "country") for k in SOURCE_FILES}
    n = {k: len(v) for k, v in ids.items()}
    gt_s1 = _load_cache(cache_dir, "gt", "s1")
    pair_row = _load_cache(cache_dir, "gt", "pair_row").astype(np.int64)
    pair_t = _load_cache(cache_dir, "gt", "pair_t")
    pair_src = _load_cache(cache_dir, "gt", "pair_src")
    log(f"  loaded cache: S1={n['s1']:,} S2={n['s2']:,} S3={n['s3']:,} GT rows={len(gt_s1):,} "
        f"pairs={len(pair_t):,}")

    # --- align ground truth with source rows, with hard integrity checks -------------------------
    for k in SOURCE_FILES:
        if len(np.unique(ids[k])) != n[k]:
            raise RuntimeError(f"duplicate entity IDs in {SOURCE_FILES[k]}")
    gt_s1row = IdIndex(ids["s1"]).rows(gt_s1)
    if (gt_s1row < 0).any():
        raise RuntimeError("ground truth contains S1 IDs missing from train_source1")
    if len(gt_s1row) != n["s1"] or len(np.unique(gt_s1row)) != n["s1"]:
        raise RuntimeError("ground truth does not have exactly one row per train S1 entity")
    pair_s1row = gt_s1row[pair_row]
    trow = {}
    for s, k in ((2, "s2"), (3, "s3")):
        m = pair_src == s
        r = IdIndex(ids[k]).rows(pair_t[m])
        if (r < 0).any():
            raise RuntimeError(f"ground truth lists {int((r < 0).sum())} {k.upper()} IDs missing from the source")
        if len(np.unique(r)) != len(r):
            raise RuntimeError(f"some {k.upper()} records are targets of more than one S1 entity — "
                               "fold inheritance would be ambiguous")
        trow[k] = (m, r)

    # --- 1. S1 split ------------------------------------------------------------------------------
    n2c = np.bincount(pair_s1row[pair_src == 2], minlength=n["s1"])
    n3c = np.bincount(pair_s1row[pair_src == 3], minlength=n["s1"])
    total = n2c + n3c
    _, ccode = np.unique(country["s1"], return_inverse=True)
    strata = (ccode.astype(np.int64) * 1000 + np.minimum(total, COUNT_CAP) * 10
              + (n2c > 0) * 2 + (n3c > 0))
    s1_keys = seeded_hash([f"S1-{int(x)}" for x in ids["s1"]], seed)
    val_s1 = stratified_select(strata, s1_keys)
    fold = {"s1": val_s1.astype(np.int8)}
    log(f"  S1 split: val={int(val_s1.sum()):,} fit={int((~val_s1).sum()):,} "
        f"({len(np.unique(strata))} strata)")

    # --- 2./3. S2, S3: targets inherit, distractors stratified by country ------------------------
    pool_stats = {}
    for k in ("s2", "s3"):
        f = np.full(n[k], -1, dtype=np.int8)
        m, r = trow[k]
        f[r] = val_s1[pair_s1row[m]].astype(np.int8)
        distract = np.flatnonzero(f < 0)
        _, dcode = np.unique(country[k][distract], return_inverse=True)
        dkeys = seeded_hash([f"{k.upper()}-{int(x)}" for x in ids[k][distract]], seed)
        dsel = stratified_select(dcode, dkeys)
        f[distract] = dsel.astype(np.int8)
        is_target = np.zeros(n[k], dtype=bool)
        is_target[r] = True
        fold[k] = f
        pool_stats[k] = {
            "val_targets": int(((f == 1) & is_target).sum()),
            "val_distractors": int(((f == 1) & ~is_target).sum()),
            "fit_targets": int(((f == 0) & is_target).sum()),
            "fit_distractors": int(((f == 0) & ~is_target).sum()),
        }
        pool_stats[k]["val_pool"] = pool_stats[k]["val_targets"] + pool_stats[k]["val_distractors"]
        pool_stats[k]["fit_pool"] = pool_stats[k]["fit_targets"] + pool_stats[k]["fit_distractors"]
        log(f"  {k.upper()}: val pool={pool_stats[k]['val_pool']:,} "
            f"(targets {pool_stats[k]['val_targets']:,} + distractors {pool_stats[k]['val_distractors']:,}), "
            f"fit pool={pool_stats[k]['fit_pool']:,}")
    for k, a in fold.items():
        np.save(out_dir / "assignment" / f"{k}_fold.npy", a)

    # --- write validation files (byte-identical rows) ----------------------------------------------
    country_stats = {}
    vdir = out_dir / VAL_DIR_NAME
    for k, fname in SOURCE_FILES.items():
        _copy_selected(data_dir / fname, vdir / f"test_source{k[-1]}.tsv", fold[k], ids[k],
                       f"S{k[-1]}-", k, country_stats)
        log(f"  wrote {VAL_DIR_NAME}/test_source{k[-1]}.tsv")
    gt_fold = val_s1[gt_s1row]
    with open(data_dir / GT_FILE, "rb") as fi, open(out_dir / VAL_GT_NAME, "wb") as fo:
        header = fi.readline()
        fo.write(header if header.endswith(b"\n") else header + b"\n")
        for row, b in _iter_data_lines(data_dir / GT_FILE):
            eid = b.split(b"\t", 1)[0].strip()
            if int(eid[3:]) != int(gt_s1[row]):
                raise RuntimeError(f"ground truth row {row} does not match the EDA cache")
            if gt_fold[row]:
                fo.write(b if b.endswith(b"\n") else b + b"\n")
    log(f"  wrote {VAL_GT_NAME}")

    # --- statistics --------------------------------------------------------------------------------
    def dist(mask):
        tc = total[mask]
        return {
            "s1_entities": int(mask.sum()),
            "singletons": int((tc == 0).sum()),
            "pct_singletons": 100.0 * float((tc == 0).mean()),
            "true_pairs": int(tc.sum()),
            "mean_matches": float(tc.mean()),
            "match_count_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(tc)) if v},
            "with_s2_and_s3": int(((n2c > 0) & (n3c > 0) & mask).sum()),
            "only_s2": int(((n2c > 0) & (n3c == 0) & mask).sum()),
            "only_s3": int(((n2c == 0) & (n3c > 0) & mask).sum()),
        }

    stats = {
        "all": dist(np.ones(n["s1"], dtype=bool)),
        "fit": dist(~val_s1),
        "val": dist(val_s1),
    }
    # largest per-stratum deviation from exact proportionality (in entities)
    uniq, inv = np.unique(strata, return_inverse=True)
    st_n = np.bincount(inv)
    st_v = np.bincount(inv, weights=val_s1)
    max_dev = float(np.max(np.abs(st_v - st_n * VAL_NUM / VAL_DEN)))

    outputs = [out_dir / VAL_GT_NAME] + [vdir / f"test_source{i}.tsv" for i in (1, 2, 3)] + \
              [out_dir / "assignment" / f"{k}_fold.npy" for k in SOURCE_FILES]
    manifest = {
        "description": "Leakage-safe 80/20 split of the TRAINING data at the Source-1 entity level. "
                       "val_as_test_dir mimics dataset/test layout but contains training records only.",
        "seed": seed,
        "validation_fraction": f"{VAL_NUM}/{VAL_DEN}",
        "s1_strata": "country x min(n_true_matches, 8) x (has_S2_match, has_S3_match)",
        "distractor_strata": "source x country",
        "ordering": "blake2b('<seed>:<entity_id>', digest_size=8), ascending, within stratum",
        "quota_rounding": "floor(n/5) + largest remainder; total = round(N/5)",
        "n_strata_s1": int(len(uniq)),
        "max_abs_stratum_deviation_entities": max_dev,
        "inputs": {fname: _stamp(data_dir / fname) for fname in list(SOURCE_FILES.values()) + [GT_FILE]},
        "counts": stats,
        "pools": pool_stats,
        "countries": country_stats,
        "outputs_sha256": {str(p.relative_to(out_dir)).replace("\\", "/"): _sha256(p) for p in outputs},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"  manifest written ({manifest['build_seconds']}s)")
    return manifest


# ------------------------------------------------------------------------------------------------
# Independent verification (reads only the written files + original train files, not the cache)
# ------------------------------------------------------------------------------------------------

def verify(data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, run_validator=True, log=print):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    vdir = out_dir / VAL_DIR_NAME
    res = {}

    # validation S1 IDs: unique and all present in train_source1
    val_s1_list = []
    with open(vdir / "test_source1.tsv", encoding="utf-8") as f:
        next(f)
        val_s1_list = [line.split("\t", 1)[0].strip() for line in f if line.strip()]
    val_s1 = set(val_s1_list)
    res["val_s1_rows"] = len(val_s1_list)
    res["val_s1_unique"] = len(val_s1) == len(val_s1_list)
    train_s1_total = 0
    found = 0
    with open(data_dir / SOURCE_FILES["s1"], encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                train_s1_total += 1
                found += line.split("\t", 1)[0].strip() in val_s1
    res["train_s1_total"] = train_s1_total
    res["val_s1_found_in_train_source1"] = found
    res["fit_s1"] = train_s1_total - found

    # validation pool: unique IDs with the right prefix per file
    pool = set()
    for i in (2, 3):
        rows = 0
        ids_i = set()
        with open(vdir / f"test_source{i}.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                if line.strip():
                    rows += 1
                    ids_i.add(line.split("\t", 1)[0].strip())
        res[f"val_pool_s{i}_rows"] = rows
        res[f"val_pool_s{i}_unique"] = len(ids_i) == rows
        res[f"val_pool_s{i}_prefix_ok"] = all(x.startswith(f"S{i}-") for x in ids_i)
        pool |= ids_i
    res["val_pool_total"] = len(pool)

    # validation ground truth: strict parse with the competition rules, every S1 present,
    # every listed target inside the validation pool
    val_gt, _ = read_id_list_tsv(out_dir / VAL_GT_NAME, GROUND_TRUTH_HEADER,
                                 required=val_s1, valid_ids=pool, strict=True)
    res["val_gt_rows"] = len(val_gt)
    val_targets = set().union(*val_gt.values()) if val_gt else set()
    res["val_targets"] = len(val_targets)
    res["val_distractors"] = len(pool - val_targets)

    # leakage: stream the ORIGINAL ground truth. Validation rows must be identical to val_gt;
    # no fit S1 may have a target inside the validation pool.
    mismatch = leaks = fit_rows = val_rows = 0
    with open(data_dir / GT_FILE, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            ids = frozenset(rest.rstrip("\n").split(",")) if rest.strip() else frozenset()
            if s1 in val_s1:
                val_rows += 1
                mismatch += val_gt.get(s1) != ids
            else:
                fit_rows += 1
                leaks += len(ids & pool)
    res["orig_gt_val_rows"] = val_rows
    res["orig_gt_fit_rows"] = fit_rows
    res["val_rows_differing_from_original_gt"] = mismatch
    res["fit_targets_inside_val_pool"] = leaks
    res["s1_in_both_folds"] = 0 if (val_rows + fit_rows == train_s1_total and found == len(val_s1)) else -1

    if run_validator:
        cmd = [sys.executable, str(VALIDATOR), "--matching", str(out_dir / VAL_GT_NAME),
               "--candidate", str(out_dir / "__no_candidate_file__.tsv"),
               "--test-dir", str(vdir), "--check-ids"]
        # the validator prints non-ASCII ("—"); force UTF-8 stdout so Windows cp1252 pipes decode
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        res["official_validator_on_val_ground_truth"] = {
            "exit_code": p.returncode, "stdout_tail": p.stdout.strip().splitlines()[-6:]}

    ok = (res["val_s1_unique"] and found == len(val_s1) and all(
        res[f"val_pool_s{i}_unique"] and res[f"val_pool_s{i}_prefix_ok"] for i in (2, 3))
        and mismatch == 0 and leaks == 0 and val_rows == len(val_s1) and res["s1_in_both_folds"] == 0
        and (not run_validator or res["official_validator_on_val_ground_truth"]["exit_code"] == 0))
    res["all_checks_passed"] = bool(ok)
    (out_dir / "verification.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"  verification: {'PASS' if ok else 'FAIL'}")
    return res


def write_report(manifest, verification, out_dir=DEFAULT_OUT_DIR):
    out_dir = Path(out_dir)
    c, p = manifest["counts"], manifest["pools"]
    L = [
        "# Validation split — construction and statistics\n",
        "Generated by `src/build_validation_split.py` (module `src/ber/validation_split.py`). "
        "Uses training data only; the test files are never read.\n",
        "## Construction\n",
        f"- **Unit of splitting: Source-1 entity.** Fraction {manifest['validation_fraction']} to validation, "
        f"seed `{manifest['seed']}`.",
        f"- S1 strata: {manifest['s1_strata']} ({manifest['n_strata_s1']} non-empty strata). "
        f"Largest per-stratum deviation from exact 20%: {manifest['max_abs_stratum_deviation_entities']:.2f} entities.",
        f"- Ordering inside a stratum: {manifest['ordering']}; quotas: {manifest['quota_rounding']}.",
        "- True targets (S2/S3 records in the ground truth) inherit the fold of their S1 entity "
        "(each target belongs to exactly one S1 — re-checked at build time).",
        f"- Distractors (S2/S3 records matched to no S1) are split 20/80 with strata {manifest['distractor_strata']}.",
        "- Result: every S1 and every S2/S3 record is in exactly one fold; fit and validation pools are disjoint.",
        f"- Validation files: `{VAL_DIR_NAME}/test_source{{1,2,3}}.tsv` (dataset/test layout, training rows only) "
        f"and `{VAL_GT_NAME}`. Fit fold: `assignment/*_fold.npy` (0 = fit, 1 = val per train-file row) + "
        "`ber.validation_split.iter_fold_lines`.\n",
        "## S1 entities\n",
        "| | all | fit | validation |",
        "|---|---:|---:|---:|",
    ]
    for key, lab in (("s1_entities", "S1 entities"), ("singletons", "singletons"),
                     ("true_pairs", "true pairs"), ("with_s2_and_s3", "matches in S2 and S3"),
                     ("only_s2", "S2 only"), ("only_s3", "S3 only")):
        L.append(f"| {lab} | {c['all'][key]:,} | {c['fit'][key]:,} | {c['val'][key]:,} |")
    L.append(f"| % singletons | {c['all']['pct_singletons']:.3f}% | {c['fit']['pct_singletons']:.3f}% | "
             f"{c['val']['pct_singletons']:.3f}% |")
    L.append(f"| mean matches / S1 | {c['all']['mean_matches']:.4f} | {c['fit']['mean_matches']:.4f} | "
             f"{c['val']['mean_matches']:.4f} |\n")
    L += ["Match-count distribution (matches per S1 → entities):\n",
          "| matches | all | fit | validation | val share |", "|---:|---:|---:|---:|---:|"]
    for k in sorted(c["all"]["match_count_distribution"], key=int):
        a = c["all"]["match_count_distribution"][k]
        v = c["val"]["match_count_distribution"].get(k, 0)
        L.append(f"| {k} | {a:,} | {c['fit']['match_count_distribution'].get(k, 0):,} | {v:,} | {100 * v / a:.2f}% |")
    L += ["", "Country distribution (from the country column of the streamed rows):\n",
          "| source | fold | country | records | share of fold |", "|---|---|---|---:|---:|"]
    for src, d in manifest["countries"].items():
        tot = defaultdict(int)
        for k, v in d.items():
            tot[k.split("|")[0]] += v
        for k, v in d.items():
            f, ctry = k.split("|")
            L.append(f"| {src.upper()} | {f} | {ctry} | {v:,} | {100 * v / tot[f]:.2f}% |")
    L += ["", "## Candidate pools (S2/S3 records)\n",
          "| source | val targets | val distractors | val pool | fit targets | fit distractors | fit pool |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in p.items():
        L.append(f"| {k.upper()} | {v['val_targets']:,} | {v['val_distractors']:,} | {v['val_pool']:,} | "
                 f"{v['fit_targets']:,} | {v['fit_distractors']:,} | {v['fit_pool']:,} |")
    vp = sum(v["val_pool"] for v in p.values())
    L.append(f"\nValidation pool: {vp:,} S2+S3 records for {c['val']['s1_entities']:,} S1 entities "
             f"= {vp / c['val']['s1_entities']:.3f} per S1 (full train: "
             f"{sum(v['val_pool'] + v['fit_pool'] for v in p.values()) / c['all']['s1_entities']:.3f}).\n")
    L += ["## Independent verification\n", "```json", json.dumps(verification, indent=1), "```\n",
          "## Known limitation\n",
          "The validation pool reproduces the TRAIN composition (4.68 S2+S3 records per S1, ~26% distractors). "
          "The test set has 5.75 S2+S3 records per S1; whether that comes from more matches per S1 or more "
          "distractors is unknowable without test labels, so validation scores may be optimistic on precision.\n"]
    (out_dir / "VALIDATION_SPLIT.md").write_text("\n".join(L), encoding="utf-8")
