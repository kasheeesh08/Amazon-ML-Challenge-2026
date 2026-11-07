# Deep France error analysis (diagnostic only)

Script: `experiments/diagnostics/france_error_analysis/france_error_analysis.py`. Raw results are in `results.json`, the log in `run.log`, and the pre-run hashes in `hashes_before.json`.

- Read-only: nothing was retrained, re-inferred, rebuilt or uploaded.
- **There are no France labels. No France F0.5 is produced, and the leaderboard is not predicted.**

## Artifacts and method

- **Labelled environments** (validation S1 and labels):
  - V: normal validation
  - T: test-ratio pool
  - S: full training pool
  - Candidates, features and probabilities come from `experiments/matching/` and `experiments/diagnostics/trank/`.
- **Models compared on labelled data:**
  - the Step 3 model, which is the same model whose test probabilities are in `experiments/test_scale/prob_test.npy`
  - the LOCO cross-country models (`experiments/diagnostics/leave_one_country_out/`)
- **Real test:** `experiments/test_scale/` (candidates, `t_*_test.npy`, Step 3 probabilities, feature stores), plus the submitted final model (`experiments/final/test_prob.npy`, `test_pred_codes.npy`).
- **Text-level probe (read-only):** raw names from `dataset/{train,test}/*_source{2,3}.tsv`, used for the initialism flag: a name that is 2–4 capital letters once spaces and punctuation are removed.
- **Like-for-like rule:** France is compared with labelled groups under the **same Step 3 model**, and with test US and India under the same test conditions. The final model's France numbers are reported separately.

## 1–2. Which labelled group does France most resemble?

| Property | V-US | T-US | S-US | T-India | TEST-US | TEST-India | **TEST-France** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidates per S1: mean / p90 / p99 | 64.2 / 141 / 214 | 65.5 / 146 / 222 | 71.0 / 164 / 249 | 51.1 / 80 / 187 | 67.1 / 145 / 225 | 77.0 / 157 / 255 | **74.6 / 173 / 292** |
| S1 with ≥ 300 candidates | 0.12% | 0.18% | 0.32% | 0.00% | 0.12% | 0.28% | **0.84%** |
| Pool records per S1 | 4.67 | 5.75 | 23.4 | 5.76 | 5.76 | 5.82 | 5.53 |
| S1 name shared by ≥ 10 / ≥ 100 S1 (same country) | 12.8% / 0.05% | 12.8% / 0.05% | same | 24.2% / 0.00% | 16.8% / 5.5% | 32.8% / 0.2% | **19.9% / 4.2%** |
| Number of S1 the name frequency is counted over | 264,728 | 264,728 | 264,728 | 176,636 | 663,106 | 809,986 | 259,452 |
| S1 with a house number | 99.48% | 99.48% | 99.48% | 90.30% | 99.47% | 90.17% | **99.51%** |
| Pool: address empty | 3.59% | 3.59% | 3.59% | 2.95% | 2.89% | 2.37% | 3.00% |
| Pool: non-Latin names | 0% | 0% | 0% | 18.3% | 0% | 18.4% | **0%** |
| Pool: postal code present | 2.8% | 2.8% | 2.8% | 2.0% | 2.9% | 2.0% | **0.27%** |
| Pool: house number present | 89.3% | 89.3% | 89.3% | 89.6% | 91.0% | 91.3% | 93.2% |
| Pool: name is an initialism | 0.11% | 0.11% | 0.12% | 0.26% | 0.10% | 0.20% | **1.74%** |
| Candidates with `t_rank` = 0 | 7.07% | 8.30% | 20.9% | 10.5% | 8.33% | 7.18% | 7.23% |
| Median `t_ncand` | 30 | 26 | 10 | 19 | 31 | 47 | 31 |

Step 3 model predictions:

| Property | V-US | T-US | S-US | T-India | TEST-US | TEST-India | **TEST-France** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Predictions per S1 | 3.353 | 3.381 | 3.830 | 3.306 | 3.618 | 3.533 | **3.829** |
| S1 with no prediction | 5.58% | 5.37% | 2.81% | 5.63% | 3.52% | 4.14% | **2.68%** |
| Predictions at p ≥ 0.95 | 90.9% | 90.4% | 82.4% | 85.1% | 85.1% | 79.3% | 87.6% |
| Predictions: house-number conflict | 14.3% | 14.4% | 16.4% | 19.7% | 19.4% | 24.1% | **9.3%** |
| Predictions: legal-form-only name difference | 24.1% | 24.2% | 25.2% | 26.5% | 26.6% | 27.5% | 27.8% |
| Predictions: `t_rank` = 0 on an uncontested target | 10.0% | 13.0% | **41.7%** | 14.0% | 16.3% | 14.5% | **10.2%** |
| Predictions: target address empty | 3.2% | 3.4% | 5.8% | 2.9% | 2.8% | 2.1% | 2.4% |
| Predictions: initialism target | 0.14% | 0.15% | 0.17% | 0.37% | 0.14% | 0.32% | **1.76%** |

- **Closest labelled analogue: US, in environment T** (validation US S1, test-like pool ratio).
- France matches the US on:
  - Latin script
  - S1 house-number presence (99.5%)
  - pool address availability
  - the share of candidates with `t_rank` = 0 and median `t_ncand` (on test)
  - the confidence profile of its predictions
- It clearly does **not** resemble S: it has the fewest uncontested predictions of any group (10.2% against 41.7% in S).
- **Three properties fall outside every labelled group:**
  1. Extreme name genericity at a comparable population size: 4.2% of France S1 share their name with 100+ others, against 0.05% of US validation S1 (a similar-sized population). Examples: "Bordeaux Club SAS" ×516, "Nantes Club" ×473.
  2. **Initialism names** (e.g. "SC" for "Solidarite Cie"): 1.74% of France pool records, 9–17× the US and India rates. They show up in 1.8% of France predictions against 0.14% of US predictions.
  3. Far fewer **house-number conflicts** in its predictions (9.3% under the Step 3 model, 3.2% under the final model), even though France has house numbers on 99.5% of S1 and 93.2% of pool records.
- Postal codes are almost absent in France (0.27%). This matters little; the postal channels carry about 0.6% of recall.

## 3. Metrics on the labelled analogue (Step 3 model; T-US unless noted)

| Metric | V-US | **T-US** | S-US | T-India | Over-merge analogue T (India-only model on US) |
|---|---:|---:|---:|---:|---:|
| Macro F0.5 | 0.97119 | **0.96426** | 0.86234 | 0.93942 | 0.95002 |
| Pair precision / recall | 98.52% / 95.50% | 97.74% / 95.55% | 85.34% / 94.50% | 96.41% / 91.99% | 96.07% / 95.99% |
| Entity precision / recall | 98.32% / 95.49% | 97.45% / 95.54% | 85.82% / 94.55% | 96.12% / 91.91% | 95.52% / 96.01% |
| FP / FN | 13,093 / 41,194 | 20,219 / 40,783 | 148,638 / 50,334 | 20,967 / 49,008 | 35,995 / 36,696 |
| FN from candidates never generated | 10,544 | 12,181 | 32,366 | 18,505 | 12,181 |
| Candidate recall | 98.85% | 98.67% | 96.47% | 96.98% | 98.67% |
| Singleton F0.5 / false-merge rate | 92.96% / 7.04% | 89.38% / 10.62% | 45.43% / 54.57% | 85.41% / 14.59% | 79.38% / 20.62% |
| F0.5: generic-name S1 (≥ 10) / unique | 0.9619 / 0.9762 | 0.9546 / 0.9696 | 0.8416 / 0.8716 | 0.9084 / 0.9563 | 0.9405 / 0.9551 |
| FP at p ≥ 0.95 (share of FP) | 1,585 (12.1%) | 2,715 (13.4%) | 27,934 (18.8%) | 3,912 (18.7%) | 7,853 (21.8%) |
| Precision: house-conflict predictions | 94.4% | 92.6% | 68.9% | 92.9% | **84.6%** |
| Precision: empty-address targets | 88.9% | 83.7% | 44.4% | 84.9% | 81.2% |
| Precision: initialism targets | 98.0% | 95.7% | 72.7% | 96.3% | 95.5% |
| Precision: legal-form-only name difference | 97.6% | 96.7% | 82.8% | 97.6% | **92.9%** |

**Recall for empty-address targets (T-US):** 87.3% reach the candidate set, but only 59.5% end up predicted.

**France-profile reweighting of US validation S1:** the US S1 are reweighted so their mix of name-frequency bucket × house presence matches France.

| Env | US, unweighted | US, France-profile weighted |
|---|---:|---:|
| V | 0.9712 | 0.9690 |
| T | 0.9643 | 0.9613 |
| S | 0.8623 | 0.8566 |

- The very generic bucket (100+) has France share 4.2% against US-validation 0.05%. It is represented by very few US S1 (weight about 92×), so the weighted value is noisy.
- 0.01% of France S1 fall in a cell with no US counterpart.

**Precision implied by the mix of prediction categories:** each prediction is assigned to a cell (probability bin × house conflict × legal-only × uncontested × address empty × initialism). The source group's per-cell precision is then applied to the target group's cell mix.

Validation of the estimator on known labelled cases:

| Case | Estimated precision | Actual | Error |
|---|---:|---:|---:|
| India in-country cells → India-only model on US (over-merge) | 0.9703 | 0.9607 | +0.0096 |
| US in-country cells → US-only model on India (under-merge) | 0.9590 | 0.9551 | +0.0039 |
| T-US cells → T-India (same model, other country) | 0.9662 | 0.9641 | +0.0021 |
| T-US cells → S-US (orphan-heavy) | 0.9558 | 0.8534 | **+0.1024** |

- Across countries the estimator is good to about 1 point of precision, and it always **overestimates**, because it cannot see degradation within a cell.
- It **fails for orphan-driven shifts**.
- Applied to France with T-US cells: **0.969**. With T-India cells: 0.969. The same cells on TEST-US give 0.965, and on TEST-India 0.952.
- **France's mix of prediction categories does not imply lower precision than the US.** Given the validation above, any France precision problem would have to lie *within* categories (cross-country, about ≥ 1 pp) or be orphan-like. The estimator cannot detect either.

## 5. Actual France predictions (unlabelled)

| | Submitted final model | Step 3 model |
|---|---:|---:|
| France S1 | 259,452 | 259,452 |
| Matches / per S1 | 912,688 / 3.518 | 993,383 / 3.829 |
| S1 with no match | 11,981 (4.62%) | 6,955 (2.68%) |
| No match: generic-name (≥ 10) / unique S1 | 4.37% / 4.65% | 2.78% / 2.54% |
| Candidates per S1: p1 / p5 / p50 / p90 / p95 / p99 / max | 30 / 34 / 43 / 173 / 215 / 292 / 533 | same |
| Probability: [0.69,0.80) / [0.80,0.90) / [0.90,0.95) / ≥ 0.95 | 2.59% / 3.62% / 2.79% / **91.00%** | 4.36% / 4.75% / 3.24% / 87.64% |
| S1 name generic (≥ 10 / ≥ 100) | 19.9% / 4.2% | same |
| Predictions: target address empty | 1.89% | 2.45% |
| Predictions: house conflict / house absent on either side | 3.18% / 7.33% | 9.26% / 7.60% |
| Predictions: initialism target | 1.82% | 1.76% |
| Predictions `t_rank`: mean / p99 | 0.033 / 0 | 0.072 / 1 |
| Predictions `t_gap`: mean / p99 | 0.0004 / 0 | 0.0008 / 0.024 |
| Predictions `t_ncand`: p5 / p50 / p90 / p99 | 1 / 7 / 39 / 86 | 1 / 7 / 38 / 85 |
| Predictions with `t_rank` = 0 on an uncontested target | 10.3% | 10.2% |

Categories that stand out:
- **Generic-name S1 are not left empty more often than unique ones.** In US and India validation, generic names do worse (T-India F0.5 0.908 generic vs 0.956 unique).
- **Initialism targets are about 13× more common than in US predictions.** Their labelled precision is high (95.7%), but it is based on very few training examples.
- **The heavy candidate tail on generic names.**

## 6. Classification

**D, not resolvable from the available evidence, with weak indications of mild over-merging (B).**

- **For B:**
  - Relative to *test* US and India, France has fewer S1 with no match (Step 3: 2.68% vs 3.52% / 4.14%; final: 4.62% vs 6.13% / 7.20%) and more predictions per S1.
  - Generic-name S1 are not left empty more often, unlike in labelled data.
  - These match the LOCO over-merge signature.
- **Against B:**
  - France's category mix is at least as clean as the US's: fewest house conflicts, high confidence, mix-implied precision 0.969.
  - Its target-context profile matches T, not the orphan-heavy S.
  - Its share of no-match S1 relative to test US (×0.76) is similar to the ratio of test US to labelled T-US (3.52% / 5.37% = ×0.66). A large part of the "fewer empty S1" pattern is **test-wide**, not France-specific.
- **For A (under-merging):** no direct evidence. The initialism-name pattern is a plausible source of *missed* France matches, since the names share no token, but test recall cannot be measured.
- **C (calibrated like the labelled data):** not demonstrable either.

## 7. Scenario range (not a prediction)

- Overall score = 0.850 × non-France + 0.150 × France.
- Non-France = 0.9442 (the blocking-gap report's estimate: test-ratio matcher behaviour, test-sized blocking losses and test country mix).

| France scenario | France value used | Implied overall |
|---|---:|---:|
| Like US, France-profile weighted (T 0.9613), minus test-sized blocking cost (−0.006) | 0.955 | 0.946 |
| Plus the LOCO over-merge penalty measured in T (−0.016) | 0.939 | 0.943 |
| Plus the worst LOCO penalty measured (−0.076) | 0.879 | 0.934 |
| Value France would need for the reported 0.924 | about 0.81 | 0.924 |

- **Every scenario grounded in a labelled analogue stays above 0.924.** Reaching 0.924 through France alone would need France to be about 0.07 worse than the worst cross-country case measured.
- Alternatively, the non-France part of test may also be below its labelled estimate. Test US and India show the same fewer-empty, more-predictions pattern relative to T. That is consistent with either more true matches per S1 in test (test has 5.75 pool records per S1 against 4.68 in train) or test-wide over-merging. Which one is **UNRESOLVED**.

## 8. Is a controlled production experiment justified?

- **A France-specific decision rule is not justified.** The earlier France-threshold experiment found it a weak lever (|Δ| < 0.001), and France's category mix shows no precision deficit.
- **A France-specific feature or model change is not justified by labels.** The initialism and extreme-genericity patterns are genuinely France-distinct, but training has almost no examples (0.1–0.2% initialisms), so any change could only be heuristic and its effect can't be measured.
- **Orphan-aware matcher experiment: this is the one experiment with labelled support.**
  - The labelled data measure a test-like competition effect (V → T −0.0085). S shows the extreme form, including overconfident false positives.
  - The only anomaly common to **all** test countries (fewer empty S1, more predictions per S1 than T) matches the direction of that mechanism.
  - Unlike any France-specific change, it can be evaluated honestly on V, T and S.
  - Expected upside at the test ratio is modest (at most about 0.009), and it would not address a France-specific problem if one exists.
- **Another diagnostic cannot resolve France** without France labels. No further France-only diagnostic is recommended.

**Recommended single next experiment (diagnostic stage, no production change):**
- Orphan-aware training on the internal fit-fold set A: add fit-fold orphan records to A's training pool at the test S1-to-pool ratio, then retrain with the unchanged Step 3 protocol.
- Evaluate on V, T and S against the current model.
- **Success criterion:** T improves by at least about 0.002 with V losing no more than about 0.001, and singleton false merges in T/S drop.
- Only then consider a production retrain.

## Integrity

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/`: all 168 files hash-identical. All 34 protected files (src, configs, models, `submission_package/`, zip) are unchanged.
- Files created, all in `experiments/diagnostics/france_error_analysis/`:
  - `france_error_analysis.py`
  - `hashes_before.json`
  - `results.json`
  - `run.log`
  - `FRANCE_ERROR_ANALYSIS_REPORT.md`
- Cost: 752 s including hashing and raw-name scans. No training or inference.
