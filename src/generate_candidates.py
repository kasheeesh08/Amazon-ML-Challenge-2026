#!/usr/bin/env python3
"""Generate candidate_pairs.tsv for a directory in competition layout (blocking only — no matching).

    python src/generate_candidates.py --test-dir experiments/validation_split/val_as_test_dir \
        --config experiments/blocking/blocking_config.json \
        --features-dir experiments/blocking/features_val --out experiments/blocking/cli_val_candidate_pairs.tsv

``--test-dir`` must contain test_source1.tsv, test_source2.tsv, test_source3.tsv. Every S1 record gets a
row (possibly empty); all S2 and S3 records form the candidate pool. Features are cached in
``--features-dir`` and reused when the source files are unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber import blocking as B  # noqa: E402
from ber.features import FeatureStore, featurise_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("[1/3] featurising ...", flush=True)
    featurise_dir(args.test_dir, args.features_dir, workers=args.workers, stem="test")
    s = {k: FeatureStore(args.features_dir, f"test_s{k}") for k in (1, 2, 3)}
    q_rows = np.arange(s[1].n)
    pool_rows = {k: np.arange(s[k].n) for k in (2, 3)}
    print(f"[2/3] generating candidates for {s[1].n:,} S1 records over {s[2].n:,} S2 + {s[3].n:,} S3 ...",
          flush=True)
    union, per_channel = B.generate_candidates(config, s[1], q_rows, {2: s[2], 3: s[3]}, pool_rows)
    print(f"[3/3] writing {len(union):,} candidate pairs -> {args.out}", flush=True)
    B.write_candidate_pairs(args.out, union, s[1].entity_ids(),
                            {2: np.asarray(s[2].col("id")), 3: np.asarray(s[3].col("id"))})
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
