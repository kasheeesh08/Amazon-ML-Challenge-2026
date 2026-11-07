# Test-scale compatibility check of the Step 3 matcher (diagnostic only)

Existing pipeline, unchanged: Step 2 featurisation + `experiments/blocking/blocking_config.json`; Step 3 features + `experiments/matching/model_gbdt.npz`; spec from `selection.json`: `{'model': 'gbdt', 'n_trees': 300, 'one_to_one': True, 'best_threshold': 0.69}`. **No test labels exist or were used; nothing below is an accuracy measurement.** No matching_results.tsv or submission was written.

## 1. Candidate distribution

| | validation | test |
|---|---:|---:|
| S1 queries | 441,364 | 1,732,544 |
| pool S2 / S3 | 1,006,652 / 1,057,368 | 4,887,273 / 5,082,316 |
| S1 per pool record | 0.2138 | 0.1738 |
| candidate pairs | 25,826,498 | 126,274,337 |
| candidates/S1 mean | 58.5 | 72.9 |
| candidates/S1 p50 | 39.0 | 42.0 |
| candidates/S1 p90 | 123.0 | 153.0 |
| candidates/S1 p95 | 160.0 | 188.0 |
| candidates/S1 p99 | 208.0 | 255.0 |
| candidates/S1 max | 390.0 | 533.0 |
| S1 with zero candidates | 0 | 0 |

Test candidates per S1 by country: US: mean 67.1, p50 40, p99 225 (n=663,106); India: mean 77.0, p50 47, p99 255 (n=809,986); France: mean 74.6, p50 43, p99 292 (n=259,452)

## 2. t_rank distribution (rank of the pair's S1 among all S1 that retrieved the same target; 0 = best)

| | validation | test |
|---|---:|---:|
| mean | 21.902 | 27.747 |
| p50 | 10.000 | 13.000 |
| p75 | 25.000 | 32.000 |
| p90 | 48.000 | 60.000 |
| p95 | 74.000 | 90.000 |
| p99 | 206.000 | 256.000 |
| max | 1329.000 | 3952.000 |
| share t_rank = 0 | 7.73% | 7.60% |
| share t_rank = 1 | 6.87% | 6.43% |
| share t_rank = 2 | 6.02% | 5.43% |
| share t_rank = 3-4 | 9.95% | 8.68% |
| share t_rank = 5-9 | 17.41% | 14.69% |
| share t_rank = 10+ | 52.02% | 57.17% |

PSI(t_rank) = 0.0297; PSI(t_ncand) = 0.0709; PSI(t_gap) = 0.0719 (PSI < 0.1 negligible, 0.1–0.25 moderate, > 0.25 major).

## 3. Matcher inference on test (no labels — counts only)

| | validation (reference) | test |
|---|---:|---:|
| pairs scored | 25,826,498 | 126,274,337 |
| predicted before one-to-one rule | 1,465,359 | 6,311,401 |
| removed by one-to-one rule | 2,138 | 57,104 |
| targets with >1 S1 before rule | 1,995 | 38,331 |
| predicted positive (final) | 1,463,221 | 6,254,297 |
| S1 with zero predictions | 25,312 | 63,811 |
| S1 with 2+ predictions | 383,826 | 1,545,794 |
| predicted S2 / S3 | 705,805 / 757,416 | 3,040,162 / 3,214,135 |
| predicted-positive rate | 5.666% | 4.953% |
| predictions per S1 (mean) | 3.315 | 3.610 |
| S1 zero-prediction rate | 5.73% | 3.68% |
| S1 2+ prediction rate | 86.96% | 89.22% |

Predictions per S1 (count → S1): validation 0:25,312, 1:32,226, 2:81,097, 3:107,273, 4:92,789, 5:59,195, 6:29,020, 7:10,731, 8:2,977, 9:657; test 0:63,811, 1:122,939, 2:269,759, 3:391,242, 4:377,087, 5:268,860, 6:146,252, 7:63,185, 8:21,526, 9:6,076

By country:

| country | set | S1 | mean predictions/S1 | zero-prediction rate |
|---|---|---:|---:|---:|
| US | validation | 264,728 | 3.353 | 5.58% |
| India | validation | 176,636 | 3.259 | 5.96% |
| US | test | 663,106 | 3.618 | 3.52% |
| India | test | 809,986 | 3.533 | 4.14% |
| France | test | 259,452 | 3.829 | 2.68% |

Validation truth for reference (Step 1/3): 5.585% singletons, mean 3.461 true matches per S1.

Probability histogram (20 bins, share of pairs): validation 93.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 5.0 | test 92.6, 0.6, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 4.1

## 4. Feature distribution shift (validation vs test)

Means over all pairs; medians / p90 / PSI on a deterministic 1% pair sample.

| feature | val mean | test mean | val p50 | test p50 | val p90 | test p90 | PSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| `n_cand_all` | 88.5063 | 113.5759 | 59.000 | 105.000 | 183.000 | 210.000 | 0.2202 moderate |
| `n_cand_src` | 45.2849 | 58.5827 | 30.000 | 53.000 | 96.000 | 111.000 | 0.1860 moderate |
| `nameaddr_cos` | 0.2898 | 0.3491 | 0.207 | 0.297 | 0.589 | 0.649 | 0.1631 moderate |
| `addr_cos` | 0.3051 | 0.2908 | 0.353 | 0.247 | 0.571 | 0.654 | 0.1547 moderate |
| `name_cont_t` | 0.3114 | 0.4270 | 0.040 | 0.316 | 1.000 | 1.000 | 0.1262 moderate |
| `name_jacc` | 0.2629 | 0.3807 | 0.021 | 0.189 | 1.000 | 1.000 | 0.1231 moderate |
| `name_cos` | 0.3084 | 0.4326 | 0.020 | 0.353 | 1.000 | 1.000 | 0.1165 moderate |
| `eq_name_n2` | 0.1687 | 0.2897 | 0.000 | 0.000 | 1.000 | 1.000 | 0.0848 |
| `eq_name_n3` | 0.1742 | 0.2962 | 0.000 | 0.000 | 1.000 | 1.000 | 0.0845 |
| `rank_all` | 43.7532 | 56.2879 | 29.000 | 37.000 | 109.900 | 130.000 | 0.0742 |
| `rank_src` | 22.1424 | 28.7914 | 14.000 | 19.000 | 57.000 | 68.000 | 0.0723 |
| `t_gap` | 0.4467 | 0.4245 | 0.474 | 0.449 | 0.811 | 0.752 | 0.0719 |
| `gap_src` | 0.5359 | 0.5001 | 0.585 | 0.525 | 0.844 | 0.804 | 0.0712 |
| `t_ncand` | 44.8033 | 56.4938 | 28.000 | 36.000 | 86.000 | 107.000 | 0.0709 |
| `addr_jacc` | 0.1810 | 0.1935 | 0.154 | 0.150 | 0.375 | 0.500 | 0.0513 |
| `addr_cont_t` | 0.2902 | 0.3094 | 0.286 | 0.286 | 0.625 | 0.714 | 0.0453 |
| `t_rank` | 21.9016 | 27.7469 | 10.000 | 13.000 | 48.000 | 60.000 | 0.0297 |
| `eq_addr_digits` | 0.4389 | 0.3621 | 0.000 | 0.000 | 1.000 | 1.000 | 0.0252 |
| `eq_house` | 0.5428 | 0.4734 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0197 |
| `is_s3` | 0.5024 | 0.5260 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0025 |
| `house_both` | 0.8557 | 0.8697 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0015 |
| `eq_addr_n2` | 0.0204 | 0.0171 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| `eq_postal` | 0.0006 | 0.0006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| `t_addr_missing` | 0.0663 | 0.0467 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| `t_name_nonlatin` | 0.0495 | 0.0455 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| `t_name_domain` | 0.0281 | 0.0173 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
| `t_name_junk` | 0.0215 | 0.0167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |

## 5. Memory / runtime

- Candidate generation: 4 query chunks of 450,000 (650.4s, 815.0s, 1032.2s, 828.7s).
- Inference: 126,274,337 pairs in 1397.9s; peak working set of the inference process 8.13 GB. Phase timings: {'featurise': 316.8, 'candidates': 3331.7, 'idf': 36.0, 'cos': 991.9, 'target_context': 124.9, 'infer': 1398.8, 'decide': 12.3, 'val_reference': 35.8}

## 6. Safety checks

- Model file SHA-256 at inference: `109324520a779ac8dcfcae8a1c422960c3a2fa2f3f3e9ed2b1b12aec0bd4c9e2`; spec {'model': 'gbdt', 'n_trees': 300, 'one_to_one': True, 'best_threshold': 0.69} read from selection.json (not tuned).
- Validation reference recomputed with the same code reproduces the saved Step 3 selection: True.
- Test directory contains only test_source{1,2,3}.tsv — no labels exist; no external data; no matching_results.tsv written.