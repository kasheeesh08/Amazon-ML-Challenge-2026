# Global threshold sensitivity (final diagnostic)

Script: `experiments/diagnostics/global_threshold/global_threshold.py`. Raw numbers are in `results.json`, the log in `run.log`.

**Nothing in production was modified.** There was no retraining, no inference, no new submission and no upload.

## Method

- **Test (unlabelled):** the exact final rule (probability ≥ t, one-S1-per-target) was re-applied to the **saved final-model probabilities** (`experiments/final/test_prob.npy` over `experiments/test_scale/cand_test.npy`).
  - At t = 0.69 it reproduces the submitted `test_pred_codes.npy` exactly.
  - At every higher threshold, re-applying the rule gives exactly the same result as filtering the submitted predictions, and the one-per-target rule holds.
  - Removed predictions were profiled with the saved target-context arrays and the unchanged Step 3 feature code (house conflict, legal form) over the test IDF.
- **Labelled environments:** the same global thresholds were applied to the **saved Step 3-model probabilities** in V (normal validation), T (test-ratio; its target-context profile matches the real test) and S (full-pool stress, orphan-heavy).
  - The final model can't be evaluated on labels it was trained on. The labelled results are therefore a proxy for how the final model would respond.

## 1–6, 10. Real test: effect on the submitted predictions (final model)

| | 0.69 (current) | 0.70 | 0.72 | 0.75 | 0.78 | 0.80 |
|---|---:|---:|---:|---:|---:|---:|
| Total predictions | 5,503,470 | 5,483,773 | 5,444,249 | 5,382,517 | 5,317,344 | 5,271,154 |
| Predictions per S1 | 3.177 | 3.165 | 3.142 | 3.107 | 3.069 | 3.042 |
| S1 with no prediction | 110,945 (6.40%) | 112,055 (6.47%) | 114,353 (6.60%) | 117,927 (6.81%) | 121,762 (7.03%) | 124,485 (7.19%) |
| Removed vs 0.69 | — | 19,697 (0.36%) | 59,221 (1.08%) | 120,953 (2.20%) | 186,126 (3.38%) | 232,316 (4.22%) |
| US removed (share of US predictions) | — | 5,693 (0.27%) | 17,107 (0.80%) | 34,893 (1.64%) | 53,539 (2.52%) | 66,856 (3.14%) |
| India removed | — | 11,897 (0.48%) | 35,896 (1.46%) | 73,606 (2.99%) | 113,516 (4.61%) | 141,777 (5.75%) |
| France removed | — | 2,107 (0.23%) | 6,218 (0.68%) | 12,454 (1.36%) | 19,071 (2.09%) | 23,683 (2.59%) |
| S1 with no prediction: US / India / France | 6.13 / 7.20 / 4.62% | 6.17 / 7.29 / 4.65% | 6.25 / 7.49 / 4.71% | 6.37 / 7.80 / 4.80% | 6.50 / 8.14 / 4.89% | 6.58 / 8.39 / 4.96% |

Removed predictions by probability band (each band is contained in the higher thresholds):

| Band | Removed predictions |
|---|---:|
| [0.69, 0.70) | 19,697 |
| [0.70, 0.72) | 39,524 |
| [0.72, 0.75) | 61,732 |
| [0.75, 0.78) | 65,173 |
| [0.78, 0.80) | 46,190 |

**Profile of removed predictions (7–9)**, against the whole submission:
- **House-number conflicts:** 52.6% of those removed at 0.70 and 46.1% at 0.80, against 12.5% of all predictions. Removal rate for house-conflict predictions vs the rest: 4.4% vs 0.6% at 0.72, and 15.6% vs 2.6% at 0.80.
- **Legal-form-only name difference:** 37.3% → 32.6% of those removed.
- **Generic names:** removed at the same rate as unique names (1.11% vs 1.07% at 0.72), so the threshold does **not** target generic-name collisions. S1 whose name is shared by 10+ others: 23.4–23.9% of removed, against 22.9% overall.
- **Target context:** 97.6–97.8% of removed predictions have `t_rank` = 0; 15.6–15.8% are on uncontested targets; median `t_ncand` is 5. Removal does not change the target-context profile.
- **Country:** India loses about 2× the share US does, and France the least.
- **Test vs labelled:** at the same threshold the test loses **about 1.5× the share** that T loses (0.72: 1.08% vs 0.71%; 0.75: 2.20% vs 1.49%). This matches the audit's finding that the test has more uncertain-band predictions.

## Labelled environments (Step 3 model; saved probabilities)

| Env | Metric | 0.69 | 0.70 | 0.72 | 0.75 | 0.78 | 0.80 |
|---|---|---:|---:|---:|---:|---:|---:|
| **V** | Macro F0.5 | **0.96285** | 0.96291 | 0.96288 | 0.96258 | 0.96185 | 0.96113 |
| V | Change vs 0.69 | — | +0.00006 | +0.00003 | −0.00027 | −0.00099 | −0.00171 |
| V | Precision / recall | 98.22 / 94.08% | 98.28 / 93.94% | 98.40 / 93.66% | 98.58 / 93.16% | 98.76 / 92.56% | 98.88 / 92.10% |
| V | FP / FN | 26,047 / 90,476 | 25,102 / 92,530 | 23,217 / 96,910 | 20,481 / 104,556 | 17,770 / 113,607 | 15,927 / 120,641 |
| V | Singleton F0.5 | 91.93% | 92.29% | 92.86% | 93.77% | 94.61% | 95.28% |
| V | Removed predictions that were true matches | — | 68.5% | 69.5% | 71.7% | 73.6% | 74.9% |
| **T** | Macro F0.5 | **0.95432** | 0.95468 | 0.95521 | 0.95572 | **0.95587** | 0.95576 |
| T | Change vs 0.69 | — | +0.00036 | +0.00089 | +0.00140 | +0.00155 | +0.00144 |
| T | US / India | 0.9643 / 0.9394 | 0.9646 / 0.9398 | 0.9652 / 0.9402 | 0.9658 / 0.9406 | 0.9662 / 0.9404 | 0.9662 / 0.9401 |
| T | Precision / recall | 97.22 / 94.12% | 97.31 / 93.99% | 97.49 / 93.71% | 97.74 / 93.23% | 98.01 / 92.65% | 98.19 / 92.22% |
| T | FP / FN | 41,186 / 89,791 | 39,745 / 91,787 | 36,903 / 96,058 | 32,866 / 103,487 | 28,752 / 112,282 | 26,043 / 118,899 |
| T | FP on orphaned targets | 9,896 | 9,596 | 9,003 | 8,179 | 7,339 | 6,781 |
| T | Singleton F0.5 | 87.79% | 88.27% | 89.16% | 90.47% | 91.79% | 92.66% |
| T | Removed predictions that were true matches | — | 58.1% | 59.4% | 62.2% | 64.4% | 65.8% |
| **S** | Macro F0.5 | **0.84002** | 0.84264 | 0.84820 | 0.85669 | 0.86553 | 0.87168 |
| S | Change vs 0.69 | — | +0.0026 | +0.0082 | +0.0167 | +0.0255 | +0.0317 |
| S | Singleton F0.5 | 42.53% | 43.69% | 46.18% | 50.28% | 54.59% | 57.67% |
| S | FP on orphaned targets | 183,185 | 178,968 | 170,283 | 156,808 | 142,786 | 133,022 |
| S | Removed predictions that were true matches | — | 14.7% | 15.0% | 16.2% | 17.5% | 18.4% |

**Consistency:**
- In **T**, the environment whose target-context profile matches the real test, raising the threshold helps monotonically up to 0.78. The gain is positive in both US and India at every threshold, and it comes from fewer false positives, including orphan false positives and singleton false merges.
- In **V** it is neutral up to 0.72, then turns slightly negative.
- In **S** it helps strongly, but S is an orphan-heavy extreme and the real test does not resemble it.
- The labelled break-even is roughly this: removal helps when **fewer than about 65–70% of the removed predictions are true matches**. That share is 58–66% in T, 68–75% in V and 15–18% in S. **It is unknown for test.**

## Scenario range for the overall effect (not a prediction)

Method: the labelled change in F0.5 per share of predictions removed, scaled linearly by the share actually removed on test. This is crude, and it uses Step 3-model behaviour as a proxy for the final model.

| Threshold | Test removed | If test behaves like V | If like T | If like S (extreme) |
|---|---:|---:|---:|---:|
| 0.70 | 0.36% | +0.0001 | +0.0006 | +0.0018 |
| 0.72 | 1.08% | +0.0001 | +0.0014 | +0.0055 |
| 0.75 | 2.20% | −0.0004 | +0.0021 | +0.0110 |
| 0.78 | 3.38% | −0.0016 | +0.0022 | +0.0167 |
| 0.80 | 4.22% | −0.0026 | +0.0020 | +0.0206 |

In the test-like scenarios (V to T) the effect is **about −0.003 to +0.002**, around the 3-decimal resolution of the reported 0.924. Only the orphan-heavy S scenario gives more than +0.005, and the earlier diagnostics show the real test does not resemble S.

## Decision

Criteria: (i) a meaningful labelled FP/precision improvement without unacceptable recall/F0.5 loss, **and** (ii) a plausible reason for the benefit to transfer to the real test.

| Threshold | Labelled evidence | Transfer rationale | Expected size (test-like) | Verdict |
|---|---|---|---:|---|
| **0.69** | Baseline, chosen on A2 | — | 0 | **Keep (recommended)** |
| 0.70 | Neutral V, +0.0004 T | Weak | ≤ +0.0006 | No measurable benefit |
| 0.72 | Neutral V (+0.00003), +0.0009 T, FP −10% in T | Plausible (test has 1.5× the uncertain band) | +0.0001 to +0.0014 | Lowest-risk alternative, but the benefit is below resolution |
| 0.75 | V −0.0003, T +0.0014 | Plausible | −0.0004 to +0.0021 | Small, sign uncertain |
| 0.78 | V −0.0010, T +0.0016 | Plausible | −0.0016 to +0.0022 | Sign uncertain |
| 0.80 | V −0.0017, T +0.0014 | Plausible | −0.0026 to +0.0020 | Sign uncertain; not justified |

**Conclusion:**
- The evidence is **consistent in direction** in the test-like environment T, but **weak in magnitude**:
  - the best test-like gain is about +0.001 to +0.002
  - the removed predictions are dominated by house-number conflicts, which were 92–95% precise in the labelled reference
  - the response of the final model (as opposed to the Step 3 model) is unverified
- Criterion (i) is only marginally met, and the expected effect is at or below the leaderboard's reported resolution.
- **Keep the production threshold at 0.69.**
- If the team ever spends a spare leaderboard submission purely as a probe, 0.72 is the only threshold with no measurable validation cost (V +0.00003). It should not replace the final submission without leaderboard evidence.

## Integrity

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, byte-identical.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/model_final.npz`: `1bac911596c3d42eca3dc0c0c01660af3eb4bd6fe27f5e68299f3f4c02866755`, unchanged. All 168 files in `experiments/final/` are unchanged.
- `camelcase_submission.zip`: `93b8db37fd2c8a79d74cc80ceb8051451b546f817393b5b1278f76426ed374a8`, unchanged. All 34 protected files are unchanged.
- Files created, all in `experiments/diagnostics/global_threshold/`:
  - `global_threshold.py`
  - `results.json`
  - `run.log`
  - `GLOBAL_THRESHOLD_REPORT.md`
- Cost: 345 s. No training or inference.
