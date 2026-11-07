#!/usr/bin/env python3
"""Local scorer: macro F0.5 of a matching_results.tsv against a ground-truth TSV.

    python src/score.py --predictions output/val_matching_results.tsv \
        --ground-truth experiments/validation_split/val_ground_truth.tsv \
        --test-dir experiments/validation_split/val_as_test_dir [--check-ids] [--json report.json]

Both files are parsed with the same rules as utils/validate_submission.py; any rejection-level format
problem makes the scorer fail (exit 2) instead of producing a number, because the leaderboard would not
score such a file. ``--test-dir`` adds the required-S1 check against test_source1.tsv and a per-country
breakdown; ``--check-ids`` also rejects IDs absent from test_source2/3.tsv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber.ids_tsv import (GROUND_TRUTH_HEADER, MATCHING_HEADER, FormatError,  # noqa: E402
                         read_id_list_tsv, read_source_ids)
from ber.metrics import macro_f05  # noqa: E402


def s1_countries(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                parts = line.rstrip("\n").split("\t")
                out[parts[0].strip()] = parts[-1].strip()
    return out


def score_files(predictions, ground_truth, test_dir=None, check_ids=False, allow_missing=False):
    truth, _ = read_id_list_tsv(ground_truth, GROUND_TRUTH_HEADER, strict=True)
    required = set(truth)
    groups = None
    valid_ids = None
    if test_dir:
        s1_path = os.path.join(test_dir, "test_source1.tsv")
        test_s1 = read_source_ids(s1_path)
        if test_s1 != required:
            raise ValueError(f"ground truth S1 set ({len(required)}) differs from {s1_path} ({len(test_s1)})")
        groups = s1_countries(s1_path)
        if check_ids:
            valid_ids = read_source_ids(os.path.join(test_dir, "test_source2.tsv")) | \
                read_source_ids(os.path.join(test_dir, "test_source3.tsv"))
    pred, _ = read_id_list_tsv(predictions, MATCHING_HEADER, required=None if allow_missing else required,
                               valid_ids=valid_ids, strict=True)
    return macro_f05(pred, truth, allow_missing=allow_missing, groups=groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", "-p", required=True)
    ap.add_argument("--ground-truth", "-g", required=True)
    ap.add_argument("--test-dir", "-t", default=None)
    ap.add_argument("--check-ids", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="diagnostics only: score missing S1 rows as empty instead of rejecting")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    args = ap.parse_args()
    try:
        rep = score_files(args.predictions, args.ground_truth, args.test_dir, args.check_ids,
                          args.allow_missing)
    except (FormatError, ValueError) as e:
        print(f"REJECTED — {e}")
        return 2
    d = rep.as_dict()
    print(f"macro F0.5 = {rep.macro_f05:.6f}  over {rep.n_entities:,} S1 entities")
    for k, v in d.items():
        if k not in ("macro_f05", "by_group"):
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v:,}")
    for g, v in rep.by_group.items():
        print(f"  [{g}] n={v['n']:,} macro F0.5={v['macro_f05']:.6f}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
