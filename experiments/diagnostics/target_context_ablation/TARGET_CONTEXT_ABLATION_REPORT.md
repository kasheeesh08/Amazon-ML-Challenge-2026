# Target-context / t_rank ablation (diagnostic only)

Script: `experiments/diagnostics/target_context_ablation/ablation_analysis.py`. Raw numbers are in `results.json`, the log in `run.log`.

**What was reused and what was new:**
- **Nothing was retrained or re-inferred.** The ablation variants were already trained with the exact Step 3 protocol by the completed `t_rank` diagnostic: A1 rows, 300 trees, threshold chosen on A2 with the one-per-target rule. That diagnostic also saved their probabilities for every candidate pair in all three environments (`experiments/diagnostics/trank/`: `model_*.npz`, `select.json`, `p_<env>_<variant>.npy`).
- The current model's probabilities reproduce the saved production values exactly (`experiments/diagnostics/trank/reproduction_checks.json`).
- This analysis adds four things:
  - entity-level metrics
  - validation-vs-stress gaps
  - the target-context feature distributions, including the **real unlabelled test candidate set**
  - probability distributions

**Variants:**

| Variant | Target-context features | Threshold (chosen on A2) |
|---|---|---:|
| A: baseline | all (the current Step 3 model) | 0.69 |
| B: no `t_rank` | `t_rank` removed | 0.68 |
| C: no target context | `t_rank`, `t_gap`, `t_ncand` removed (the t_rank diagnostic calls this "B2") | 0.71 |
| D: optional stable form | `t_rank` replaced by its within-target percentile, `t_rank / max(t_ncand − 1, 1)`, which does not depend on the absolute number of competing S1 | 0.70 |
| (extra) capped | `t_rank` capped at 1 | 0.70 |

**Environments** (the 441,364 validation S1 and their labels):
- **V:** normal validation, the validation pool.
- **S:** test-scale stress, the full 10.3M training pool (0.043 S1 per pool record).
- **T:** the validation pool plus random fit-fold records, sized to the real test's 0.174 S1 per pool record.

## 1. Main comparison

| | Validation F0.5 | Stress F0.5 | Gap (V − S) | Test-ratio F0.5 | Gap (V − T) |
|---|---:|---:|---:|---:|---:|
| **Baseline (A)** | **0.96285** | **0.84002** | 0.12283 | **0.95432** | 0.00853 |
| No `t_rank` (B) | 0.96281 | 0.83753 | 0.12528 | 0.95420 | 0.00861 |
| No target context (C) | 0.95527 | **0.90154** | **0.05373** | 0.95125 | **0.00402** |
| Stable `t_rank` percentile (D) | 0.96295 | 0.84347 | 0.11948 | 0.95482 | 0.00813 |
| (extra) `t_rank` capped at 1 | 0.96292 | 0.84303 | 0.11989 | 0.95471 | 0.00821 |

## 2. Precision, recall, singletons, errors

| Variant | Env | Pair P | Pair R | Entity P | Entity R | Singleton F0.5 | Singleton false-merge rate | FP | FN | FP on singleton S1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | V | 98.22% | 94.08% | 98.01% | 94.04% | 91.93% | 8.07% | 26,047 | 90,476 | 2,110 |
| A | T | 97.22% | 94.12% | 96.92% | 94.09% | 87.79% | 12.21% | 41,186 | 89,791 | 3,316 |
| A | S | 82.45% | 93.14% | 83.69% | 93.14% | 42.53% | 57.47% | 302,910 | 104,785 | 25,968 |
| B | V | 98.20% | 94.13% | 97.98% | 94.09% | 91.77% | 8.23% | 26,361 | 89,660 | 2,158 |
| B | T | 97.19% | 94.17% | 96.89% | 94.14% | 87.65% | 12.35% | 41,585 | 89,080 | 3,348 |
| B | S | 82.16% | 93.19% | 83.39% | 93.20% | 41.36% | 58.64% | 309,067 | 103,997 | 26,574 |
| C | V | 98.67% | 90.90% | 98.49% | 90.85% | 93.64% | 6.36% | 18,681 | 139,035 | 1,643 |
| C | T | 98.28% | 90.79% | 98.04% | 90.74% | 91.73% | 8.27% | 24,308 | 140,670 | 2,167 |
| C | S | 93.22% | 89.03% | 92.65% | 89.01% | 69.32% | 30.68% | 98,948 | 167,605 | 9,428 |
| D | V | 98.32% | 93.87% | 98.12% | 93.83% | 92.45% | 7.55% | 24,534 | 93,662 | 1,983 |
| D | T | 97.35% | 93.92% | 97.07% | 93.89% | 88.41% | 11.59% | 39,033 | 92,838 | 3,129 |
| D | S | 82.87% | 93.02% | 84.14% | 93.02% | 44.12% | 55.88% | 293,643 | 106,693 | 25,091 |

FN includes true pairs that blocking never produced: 27,277 in V, 30,686 in T, 66,750 in S.

**False positives on orphan-like targets:**
- In V there are no orphans. Instead the table shows false positives on uncontested targets, i.e. `t_rank` = 0 with only one S1 retrieving the target.
- In T and S the false positives are split by where the target record comes from: fit-fold targets whose owner S1 isn't queried (true orphans), and fit-fold distractors.

| Variant | V: FP on uncontested targets | T: FP on orphans / distractors / uncontested | S: FP on orphans / distractors / uncontested |
|---|---:|---:|---:|
| A | 3,111 | 9,896 / 4,495 / 6,473 | 183,185 / 89,373 / 122,984 |
| B | 3,178 | 10,014 / 4,525 / 6,579 | 186,819 / 91,352 / 125,352 |
| C | 1,652 | 2,561 / 3,479 / 2,831 | 27,834 / 55,060 / 35,884 |
| D | 2,969 | 9,471 / 4,258 / 6,181 | 178,920 / 85,516 / 119,234 |

## 3. Feature distributions: normal validation vs stress (and the real test)

All candidate pairs. TEST is the real unlabelled test candidate set (every 5th of 126,274,337 pairs). PSI is measured against V.

| | V | T | S | TEST |
|---|---:|---:|---:|---:|
| `t_rank`: mean / p50 / p90 | 21.9 / 10 / 48 | 19.3 / 9 / 43 | 7.9 / 3 / 17 | 27.7 / 13 / 60 |
| `t_gap`: mean / p50 / p90 | 0.447 / 0.475 / 0.811 | 0.385 / 0.385 / 0.787 | 0.167 / 0.071 / 0.508 | 0.424 / 0.449 / 0.752 |
| `t_ncand`: mean / p50 / p90 | 44.8 / 27 / 86 | 39.5 / 23 / 76 | 16.9 / 10 / 32 | 56.5 / 36 / 107 |
| Share of candidates with `t_rank` = 0 | 7.73% | 9.07% | 20.80% | 7.59% |
| Share of uncontested targets (`t_ncand` = 1) | 0.87% | 1.29% | 7.54% | 1.17% |
| PSI `t_rank` / `t_gap` / `t_ncand` vs V | — | 0.012 / 0.053 / 0.024 | **0.558 / 1.175 / 0.914** | 0.031 / 0.070 / 0.072 |

**By outcome, current model A:**

| Env | Group | n | Share with `t_rank` = 0 | Share uncontested | Median `t_ncand` |
|---|---|---:|---:|---:|---:|
| V | TP | 1,437,174 | 99.84% | 10.45% | 6 |
| V | FP | 26,047 | 99.11% | 11.94% | 6 |
| V | FN (blocked) | 63,199 | 67.40% | 7.04% | 10 |
| V | TN | 24,300,078 | 2.03% | 0.27% | 29 |
| T | TP / FP | 1,437,859 / 41,186 | 99.84% / 99.05% | 13.33% / 15.72% | 5 / 5 |
| S | TP / FP | 1,422,865 / 302,910 | 99.89% / 99.30% | 39.96% / 40.60% | 2 / 2 |
| S | TN | 30,367,518 | **16.25%** | 5.67% | 11 |
| TEST | Predicted (Step 3 model; unlabelled) | 6,254,297 | 99.50% | 14.49% | 5 |

**Prediction probabilities (A vs C):**

| Env | Variant | Candidates ≥ 0.69 | Mean prob of TP | Mean prob of FP | FP at ≥ 0.95 | Mean prob of blocked FN |
|---|---|---:|---:|---:|---:|---:|
| V | A / C | 5.67% / 5.52% | 0.983 / 0.982 | 0.838 / 0.845 | 14.6% / 14.8% | 0.393 / 0.378 |
| T | A / C | 5.62% / 5.42% | 0.984 / 0.982 | 0.843 / 0.846 | 16.1% / 15.1% | 0.396 / 0.379 |
| S | A / C | 5.39% / 4.62% | 0.986 / 0.982 | 0.859 / 0.850 | **21.5%** / 15.9% | 0.417 / 0.383 |

On the real test, the Step 3 model puts 5.00% of candidates at ≥ 0.69, and 82.9% of its predictions are at ≥ 0.95.

## 4. Answers

**A. Is `t_rank` responsible for the distribution-shift problem?**
- `t_rank` alone is not. Removing it (B) changes every environment by at most 0.0025, because the model moves its reliance to `t_gap`, which is exactly 0 whenever `t_rank` = 0.
- The **target-context family as a whole** does cause the collapse in the **stress** environment. Removing all three halves the V−S gap (0.123 → 0.054) and cuts orphan false positives from 183k to 28k.
- For the **real test set**, the measured evidence points away from target context as the main cause of the leaderboard gap:
  - The real test's target-context distributions are close to V and T (PSI ≤ 0.072) and far from S (PSI 0.56–1.17).
  - Test predictions look like V and T: 99.5% have `t_rank` = 0 and 14.5% are uncontested, against 40% in S.
  - In the test-like environment T, the target-context features are a **net benefit**: A scores 0.9543 and C scores 0.9513.

**B. Genuine matching information or target-population structure?**
- Both, and they can't be separated. `t_rank` encodes a measured structural fact: no S2/S3 record belongs to more than one S1 (0 of 7.64M training pairs).
- In every environment, 99.8% of true positives and 99.1% of false positives have `t_rank` = 0. `t_rank` = 0 works as a near-**necessary** condition for predicting a match, not as something that separates true from false matches. The model uses it as a soft "is this S1 the best claimant for this target" assignment.
- That assignment is real information when every target's owner is queried: it is worth +0.0076 on V and +0.003 on T. It fails when owners are missing. In S, 16.3% of true negatives have `t_rank` = 0, against 2.0% in V.

**C. Does removing `t_rank` improve robustness in the stress environment?** No. B scores 0.8375 against A's 0.8400 in S, and 0.9542 against 0.9543 in T.

**D. Does removing all target context behave differently?** Yes.
- Precision rises in every environment (V 98.22% → 98.67%; S 82.45% → 93.22%), but recall falls sharply (V 94.08% → 90.90%; about 48.6k more false negatives).
- Net effect: +0.0615 in S, −0.0076 in V and −0.0031 in T.
- The percentile form (D) and the capped form differ from the baseline by at most 0.0035 in any environment. Both keep the `t_rank` = 0 signal, which is the damaging part.

**E. Is the model over-relying on target-competition statistics?**
- Under orphan conditions, yes:
  - The three features hold 82.8% of the GBDT gain.
  - In S, 21.5% of false positives come with probability ≥ 0.95, against 15.9% without target context.
- Under test-like conditions (T), the cost is small: the V−T gap is 0.0085, and target context remains net positive. Relying on these features is harmful only in environments where owners are missing.

**F. Recommendation:**
1. **Keep the current matcher.** Removing `t_rank` (option 2) gives no gain anywhere. Removing all target context (option 3) loses 0.0076 on V and 0.0031 on T; it wins only in S, whose feature distribution the real test does not resemble.
2. **Redesign (option 4)** is only justified as a hedge. The most direct form is orphan-aware training rather than a new feature. Its measurable upside at the test ratio is at most about the V−T gap of 0.009.
3. **Another diagnostic (option 5)** should target the remaining unexplained leaderboard gap rather than target context. After the test-ratio orphan effect (about 0.009), the country-mix shift (about 0.004) and a LOCO-sized France penalty, roughly 0.015 is still unexplained.
4. No production change is recommended on this evidence. Nothing here supports any claim about the real leaderboard score.

## 5. Computational cost

- 495 s in total: about 2 minutes per environment, the test-distribution pass, and hashing of the protected files and `experiments/final` (168 files) before and after.
- No training and no inference. The variant training (about 11 minutes) and scoring (about 30 minutes) were done earlier by the `t_rank` diagnostic and were reused.
- Peak memory was modest: columns were read in chunks from the existing feature matrices.

## 6. Integrity and files

- `output/matching_results.tsv`: SHA-256 `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: SHA-256 `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/`: all 168 files hash-identical before and after. `model_final.npz` is `1bac911596c3d42eca3dc0c0c01660af3eb4bd6fe27f5e68299f3f4c02866755`.
- The 34 protected production files (src, configs, models, `submission_package/`, zip) are unchanged.
- Files created, all in `experiments/diagnostics/target_context_ablation/`:
  - `ablation_analysis.py`
  - `hashes_before.json`
  - `results.json`
  - `run.log`
  - `TARGET_CONTEXT_ABLATION_REPORT.md`
