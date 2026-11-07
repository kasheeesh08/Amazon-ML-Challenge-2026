#!/usr/bin/env python3
"""Build (and independently verify) the leakage-safe 80/20 validation split.

    python src/build_validation_split.py                 # full training data
    python src/build_validation_split.py --verify-only   # re-run the checks on existing outputs

Smoke test on the small EDA sample:

    python src/build_validation_split.py --data-dir experiments/_smoke/dataset/train \
        --cache-dir experiments/_smoke/_eda_cache --out-dir experiments/_smoke/validation_split

Requires the EDA hash cache (experiments/eda.py). Reads training files only.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber import validation_split as vs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(vs.DEFAULT_DATA_DIR))
    ap.add_argument("--cache-dir", default=str(vs.DEFAULT_CACHE_DIR))
    ap.add_argument("--out-dir", default=str(vs.DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=vs.SEED)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-validator", action="store_true", help="skip the official-validator check")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.verify_only:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    else:
        print("[1/2] building split ...", flush=True)
        manifest = vs.build(args.data_dir, args.cache_dir, out, args.seed)
    print("[2/2] verifying ...", flush=True)
    ver = vs.verify(args.data_dir, out, run_validator=not args.skip_validator)
    vs.write_report(manifest, ver, out)
    print(f"report: {out / 'VALIDATION_SPLIT.md'}")
    return 0 if ver["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
