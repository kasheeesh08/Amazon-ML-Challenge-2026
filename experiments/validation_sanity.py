#!/usr/bin/env python3
"""Scorer sanity checks on the validation split (no blocking, no model — synthetic predictions only).

For each synthetic prediction set:
  1. write it with ber.ids_tsv.write_id_list_tsv,
  2. run the OFFICIAL utils/validate_submission.py on it (--check-ids, validation dir as --test-dir),
  3. score it with src/score.py's score_files (the path real predictions will take),
  4. compare with an expectation computed independently: exact Fractions from a separate
     implementation, plus closed forms where they exist.

Prediction sets:
  perfect         truth itself                                  -> 1.0
  all_empty       every list empty                              -> singleton fraction
  keep_one        non-singletons keep only their smallest true ID; per entity 5/(n+4)
  add_distractor  truth + one distinct pure distractor per S1   ; singleton 0, else 5n/(5n+4)
  keep_half       non-singletons keep ceil(n/2) true IDs
  rotate          each S1 receives the truth of the next S1     -> near 0 (only singleton->singleton)

Writes experiments/validation_split/sanity_results.json. Temporary prediction files are deleted.

    python experiments/validation_sanity.py [--split-dir experiments/validation_split]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ber.ids_tsv import GROUND_TRUTH_HEADER, MATCHING_HEADER, read_id_list_tsv, read_source_ids, write_id_list_tsv  # noqa: E402,E501
from score import score_files  # noqa: E402


def exact_entity(pred, true):
    """Independent exact implementation (Fractions) of the per-entity rule."""
    if not true:
        return Fr(1) if not pred else Fr(0)
    tp = len(pred & true)
    if not pred or tp == 0:
        return Fr(0)
    p, r = Fr(tp, len(pred)), Fr(tp, len(true))
    return Fr(5, 4) * p * r / (Fr(1, 4) * p + r)


def run_validator(pred_path, test_dir):
    cmd = [sys.executable, str(ROOT / "utils" / "validate_submission.py"), "--matching", str(pred_path),
           "--candidate", str(pred_path.parent / "__none__.tsv"), "--test-dir", str(test_dir), "--check-ids"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    last = [line for line in p.stdout.splitlines() if line.startswith(("PASS", "FAIL"))]
    return p.returncode, (last[-1] if last else p.stdout.strip().splitlines()[-1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default=str(ROOT / "experiments" / "validation_split"))
    args = ap.parse_args()
    split = Path(args.split_dir)
    test_dir = split / "val_as_test_dir"
    gt_path = split / "val_ground_truth.tsv"
    tmp = split / "_sanity_tmp"
    tmp.mkdir(exist_ok=True)

    truth, _ = read_id_list_tsv(gt_path, GROUND_TRUTH_HEADER, strict=True)
    order = sorted(truth)
    pool = read_source_ids(test_dir / "test_source2.tsv") | read_source_ids(test_dir / "test_source3.tsv")
    targets = set().union(*truth.values())
    distractors = sorted(pool - targets)
    n_single = sum(1 for s in order if not truth[s])
    print(f"validation S1={len(order):,}, singletons={n_single:,}, pool={len(pool):,}, "
          f"pure distractors={len(distractors):,}", flush=True)
    if len(distractors) < len(order):
        raise RuntimeError("not enough pure distractors for the add_distractor check")

    def gen(name):
        if name == "perfect":
            return {s: truth[s] for s in order}
        if name == "all_empty":
            return {s: frozenset() for s in order}
        if name == "keep_one":
            return {s: frozenset([min(truth[s])]) if truth[s] else frozenset() for s in order}
        if name == "keep_half":
            return {s: frozenset(sorted(truth[s])[:math.ceil(len(truth[s]) / 2)]) for s in order}
        if name == "add_distractor":
            return {s: truth[s] | {distractors[i]} for i, s in enumerate(order)}
        if name == "rotate":
            return {s: truth[order[(i + 1) % len(order)]] for i, s in enumerate(order)}
        raise KeyError(name)

    def closed_form(name, pred):
        if name == "perfect":
            return Fr(1)
        if name == "all_empty":
            return Fr(n_single, len(order))
        if name == "keep_one":
            return sum((Fr(5, len(truth[s]) + 4) if truth[s] else Fr(1)) for s in order) / len(order)
        if name == "add_distractor":
            return sum((Fr(5 * len(truth[s]), 5 * len(truth[s]) + 4) if truth[s] else Fr(0))
                       for s in order) / len(order)
        return None

    results = {}
    for name in ("perfect", "all_empty", "keep_one", "keep_half", "add_distractor", "rotate"):
        t0 = time.time()
        pred = gen(name)
        path = tmp / f"{name}.tsv"
        write_id_list_tsv(path, pred, MATCHING_HEADER, s1_order=order)
        vcode, vline = run_validator(path, test_dir)
        rep = score_files(path, gt_path, test_dir=str(test_dir), check_ids=True)
        exact = sum(exact_entity(set(pred[s]), set(truth[s])) for s in order) / len(order)
        cf = closed_form(name, pred)
        results[name] = {
            "official_validator_exit_code": vcode,
            "official_validator": vline,
            "scorer_macro_f05": rep.macro_f05,
            "independent_exact_macro_f05": float(exact),
            "closed_form_macro_f05": float(cf) if cf is not None else None,
            "abs_diff_vs_exact": abs(rep.macro_f05 - float(exact)),
            "f05_singletons": rep.f05_singletons,
            "f05_nonsingletons": rep.f05_nonsingletons,
            "by_country": rep.by_group,
            "seconds": round(time.time() - t0, 1),
        }
        ok = vcode == 0 and abs(rep.macro_f05 - float(exact)) < 1e-12 and \
            (cf is None or abs(float(cf) - float(exact)) < 1e-12)
        results[name]["ok"] = ok
        vword = str(vline).split()[0] if vline else "?"
        print(f"  {name:15s} validator={vword} scorer={rep.macro_f05:.6f} exact={float(exact):.6f} "
              f"closed={'-' if cf is None else f'{float(cf):.6f}'} {'OK' if ok else 'MISMATCH'}", flush=True)
        path.unlink()

    # rejection path: a missing S1 row must make the scorer refuse to score
    broken = tmp / "missing_row.tsv"
    write_id_list_tsv(broken, {s: truth[s] for s in order[1:]}, MATCHING_HEADER, s1_order=order[1:])
    try:
        score_files(broken, gt_path, test_dir=str(test_dir))
        results["missing_row_rejected"] = False
    except Exception as e:  # FormatError
        results["missing_row_rejected"] = True
        results["missing_row_message"] = str(e)[:200]
    vcode, vline = run_validator(broken, test_dir)
    results["missing_row_official_validator_exit_code"] = vcode
    shutil.rmtree(tmp)

    results["all_ok"] = all(v["ok"] for v in results.values() if isinstance(v, dict) and "ok" in v) \
        and results["missing_row_rejected"] and results["missing_row_official_validator_exit_code"] == 1
    (split / "sanity_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"all_ok={results['all_ok']} -> {split / 'sanity_results.json'}")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
