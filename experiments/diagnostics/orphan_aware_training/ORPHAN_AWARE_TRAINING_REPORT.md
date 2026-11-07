# Orphan-aware training (diagnostic only)

Script: `experiments/diagnostics/orphan_aware_training/orphan_aware.py`.

- Raw results: `evaluate.json`, `train.json`, `select.json`, `setup.json`, `candidates.json`, `threshold_curves.json`.
- Logs: `run.log`, `threshold_curves.log`. Hashes: `hashes_before.json`, `hashes_after.json`.
- **Production untouched:** the final model was not retrained, test inference was not run, the threshold was not changed, and nothing was uploaded.

## Design

- **A, baseline:** the saved Step 3 model.
  - It was trained on the internal fit-fold set A: 441,364 S1, with a pool of their own targets plus fit distractors (4.68 records per S1).
  - Every target's owner is queried, so A never sees orphaned targets.
- **B, orphan-aware:** the same protocol, but A's pool is extended with **475,494 fit-fold target records whose owner S1 is not in A** (237,674 S2 and 237,820 S3).
  - These are orphaned targets under the same definition as the stress diagnostic.
  - The pool per S1 becomes 5.754, the real test ratio (0.1738 S1 per pool record).
  - Only training-fold labels are used; 0 validation records.
  - Then, for B:
    - Candidates were regenerated with the unchanged Step 2 configuration: 26,441,206 pairs, against 25,794,241 for A.
    - IDF was taken from the extended pool, and features and target context were built with the unchanged Step 3 code.
    - Training rows are A1 S1, all positives plus the same deterministic 10% hashed negatives at weight 10: 3,192,047 rows, of which 396,367 are on orphaned targets (33,091 of those with `t_rank` = 0).
  - The GBDT uses exactly the Step 3 settings: 300 trees, depth 6, learning rate 0.1, λ 1.0, minimum child weight 20, 64 bins, seed 0.
  - **The threshold chosen on A2 (Step 3 rule) is 0.69**, the same as production, so no threshold change is involved.
- **Pipeline check:** the smoke run added no orphans, because the smoke pool is already denser than the test ratio. In that case B reproduced A's predictions exactly, which confirms the re-implemented pipeline.
- **Evaluation:** the same saved labelled environments, V (normal validation), T (test-ratio) and S (full-pool stress), with unchanged feature matrices.

## Results at the production threshold 0.69

| Metric | V: A | V: B | T: A | T: B | S: A | S: B |
|---|---:|---:|---:|---:|---:|---:|
| **Macro F0.5** | **0.96285** | 0.96172 | **0.95432** | **0.95652** | **0.84002** | **0.87194** |
| Pair precision | 98.22% | 98.52% | 97.22% | 97.88% | 82.45% | 87.74% |
| Pair recall | 94.08% | 93.08% | 94.12% | 93.19% | 93.14% | 92.52% |
| Entity precision | 98.01% | 98.33% | 96.92% | 97.62% | 83.69% | 87.65% |
| Entity recall | 94.04% | 93.03% | 94.09% | 93.15% | 93.14% | 92.53% |
| Singleton F0.5 | 91.93% | 93.13% | 87.79% | 90.27% | 42.53% | 51.73% |
| False positives | 26,047 | 21,347 | 41,186 | 30,830 | 302,910 | 197,495 |
| False negatives | 90,476 | 105,755 | 89,791 | 104,056 | 104,785 | 114,293 |
| FP on orphaned targets | 0 | 0 | 9,896 | **4,796** | 183,185 | **93,500** |
| FP on fit distractors | 0 | 0 | 4,495 | 3,922 | 89,373 | 78,515 |
| FP with `t_rank` = 0 on an uncontested target | 3,111 | 2,633 | 6,473 | 4,896 | 122,984 | 79,777 |
| FP at p ≥ 0.95 | 3,798 | 2,968 | 6,627 | 4,335 | 65,256 | 30,604 |
| Precision of `t_rank` = 0 predictions | 98.23% | 98.53% | 97.24% | 97.89% | 82.53% | 87.77% |
| Candidates at ≥ 0.69 | 5.67% | 5.59% | 5.62% | 5.52% | 5.39% | 5.02% |

**Gaps:**

| | A | B |
|---|---:|---:|
| Validation → test-ratio | 0.0085 | **0.0052** |
| Validation → stress | 0.1228 | **0.0898** |

**Changes from A to B at 0.69:**

| Env | Change |
|---|---:|
| V | **−0.0011** |
| T | **+0.0022** |
| S | **+0.0319** |

**Target-context behaviour:**
- In every environment, false positives still have `t_rank` = 0 and `t_gap` = 0 in 99.1–99.6% of cases.
- What changes is how much the model *trusts* that signal:
  - GBDT gain of the target-context features falls from 82.8% to **55.1%** (`t_rank` 76.5% → 52.6%, `t_gap` 6.1% → 2.3%).
  - Address agreement rises: `addr_jacc` from < 1% to **28.1%**, and `nameaddr_cos` from 1.6% to 3.6%.

## Better model, or just more conservative? (threshold curves, `threshold_curves.json`)

Macro F0.5 at each threshold (one-per-target rule on):

| Threshold | 0.61 | 0.67 | 0.70 | 0.73 | 0.76 | 0.79 | 0.82 | 0.85 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V / A | 0.9616 | 0.9626 | **0.9629** | 0.9628 | 0.9624 | 0.9616 | 0.9600 | 0.9579 |
| V / B | 0.9615 | **0.9618** | 0.9616 | 0.9610 | 0.9600 | 0.9585 | 0.9564 | 0.9535 |
| T / A | 0.9509 | 0.9535 | 0.9547 | 0.9554 | 0.9558 | **0.9559** | 0.9552 | 0.9540 |
| T / B | 0.9547 | 0.9562 | 0.9567 | **0.9568** | 0.9564 | 0.9555 | 0.9541 | 0.9518 |
| S / A | 0.8200 | 0.8347 | 0.8426 | 0.8510 | 0.8597 | 0.8686 | 0.8779 | 0.8872 |
| S / B | 0.8544 | 0.8674 | 0.8742 | 0.8814 | 0.8887 | 0.8958 | 0.9026 | 0.9092 |

- **Best achievable score:**
  - V: A 0.9629, B 0.9618 (B −0.0011).
  - T: A 0.9559 at 0.79, B 0.9568 at 0.73 (B **+0.0009**).
  - S: B above A at every threshold.
- **At matched validation (V ≈ 0.9616):**

  | Operating point | T | S |
  |---|---:|---:|
  | A at threshold 0.79 | 0.9559 | 0.8686 |
  | B at threshold 0.70 | 0.9567 (**+0.0008**) | 0.8742 (**+0.0056**) |

  B cannot reach A's top validation score at any threshold.
- **Most of the +0.0022 on T at 0.69 is a shift toward conservatism.** Raising the baseline's threshold from 0.69 to 0.76–0.79 already gives +0.0011–0.0016 on T, for −0.0005 to −0.0013 on V. The orphan-aware model's own improvement at the test ratio is about +0.0008–0.0009.

## Conclusions

- **Improves test-ratio without materially hurting validation?** Only marginally.
  - At 0.69: T +0.0022, V −0.0011. That just meets the pre-set T criterion and slightly exceeds the V tolerance.
  - At matched validation, T improves by only about +0.0008.
- **Mechanism:** training on orphaned targets (33k non-matches with `t_rank` = 0) teaches the model that being the best claimant for a target isn't sufficient.
  - It moves weight from target-competition statistics to address agreement.
  - That removes about half the orphan-driven false positives and 35–53% of the p ≥ 0.95 false positives, at the cost of about 15k extra false negatives.
  - This is a real robustness gain **under heavy orphan conditions** (S: +0.032 at 0.69, +0.006 at matched V). The real test's target-context profile matches T, not S (target-context ablation report).
- **No claim about the leaderboard** follows from these results.

## Worth ONE controlled submission?

- **Not on its own merits.** At the test ratio, its advantage beyond a simple threshold change is about +0.001, below the 3-decimal resolution of the reported score.
- It would also require redesigning *final* training. The final model trains on all training S1 against the full training pool, where every owner is queried, so it contains no orphans. Making it orphan-aware means deliberately leaving S1 unqueried. That is a larger change, with its own risk.
- **This family of variants should not be explored further.** The measured ceiling at the test ratio is too small.

## Single most promising next action

**No further model-side change has a measurable expected gain larger than about 0.002.**

If the team still wants exactly one controlled leaderboard submission to test the one mechanism with consistent labelled support ("be slightly more conservative under the test population"), the cheapest version needs no retraining:
- **Re-threshold the existing final-model probabilities globally**, e.g. to 0.75, and write the result to a new file without touching the current submission.
- In labelled T with the Step 3 model, this gives +0.0011–0.0016 at −0.0005–0.0013 on V.
- It transfers to the final model only approximately, and the expected leaderboard change is within about ±0.002 with uncertain sign.
- It requires your approval, and it may be invisible at 3 decimals.

Otherwise, the recommendation is to **keep the current submission**.

## Integrity and cost

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/`: all 168 files hash-identical. `model_final.npz` is `1bac911596c3d42eca3dc0c0c01660af3eb4bd6fe27f5e68299f3f4c02866755`.
- All 34 protected files (src, configs, models, `submission_package/`, zip) are unchanged.
- **Cost:** about 50 minutes in total.

  | Phase | Time |
  |---|---:|
  | Candidates | 442 s |
  | Features | 679 s |
  | Training | 510 s |
  | Threshold selection | 101 s |
  | Scoring V/T/S | 1,056 s |
  | Evaluation | 141 s |
  | Threshold curves | ~2 min |

  Peak working set was about 10 GB, including memory-mapped pages. New artifacts take about 6 GB (`X_B.npy` 5.3 GB).
- **Files created** (all in `experiments/diagnostics/orphan_aware_training/`):
  - `orphan_aware.py`
  - `poolB_s2.npy`, `poolB_s3.npy`, `cand_B.npy`, `idf_B_name.npz`, `idf_B_addr.npz`, `X_B.npy`, `y_B.npy`
  - `model_orphan_aware.npz`, `pA2_B.npy`, `p_V_B.npy`, `p_T_B.npy`, `p_S_B.npy`
  - `setup.json`, `candidates.json`, `train.json`, `select.json`, `evaluate.json`, `threshold_curves.json`, `timings.json`
  - `hashes_before.json`, `hashes_after.json`
  - `run.log`, `threshold_curves.log`
  - `ORPHAN_AWARE_TRAINING_REPORT.md`
- A smoke copy is in `experiments/_smoke/diagnostics_orphan_aware/`.
