# Final pipeline run report

Pipeline `experiments/final_pipeline.py`; fixed configuration from Steps 2–3 (nothing tuned). No test labels exist; no F0.5 is claimed for test.

## Configuration

- Blocking: `C:\Users\kashish\Desktop\Amazon-ML-Challenge-2026\experiments\blocking\blocking_config.json` (SHA-256 `91a3897685414bb48c4bc4f00a162e1b245eb8640166f950a76bdc379042e0f5`) — Step 2, unchanged.
- Features: `src/ber/match_features.py`, 50 features (Step 3, unchanged).
- Matcher: GBDT {'n_trees': 300, 'max_depth': 6, 'learning_rate': 0.1, 'lam': 1.0, 'min_child_weight': 20.0, 'seed': 0}, 300 trees; decision threshold **0.69**, one-S1-per-target rule **ON**. Final model SHA-256 `1bac911596c3d42eca3dc0c0c01660af3eb4bd6fe27f5e68299f3f4c02866755`.
- Step 3 validated artifacts matched their recorded hashes: {'experiments/matching/model_gbdt.npz': {'sha256': '109324520a779ac8dcfcae8a1c422960c3a2fa2f3f3e9ed2b1b12aec0bd4c9e2', 'matches_validated': True}, 'experiments/matching/selection.json': {'sha256': 'a5fcf91cf76b431c002e2add7821c8cce972712615227f12ce3ae3ebf2f15fc9', 'matches_validated': True}, 'experiments/blocking/blocking_config.json': {'sha256': '91a3897685414bb48c4bc4f00a162e1b245eb8640166f950a76bdc379042e0f5', 'matches_validated': True}}

## Final training data (training labels only)

- 2,206,821 training S1 × full training pool → 160,550,236 candidate pairs (candidate recall 95.646%).
- Candidate pairs: 7,305,755 positive / 153,244,481 negative. Training rows: 22,629,394 = all 7,305,755 positives + 15,323,639 sampled negatives (rate 0.1, weight 10.0).
- GBDT training time 1075.7s; memory {'peak_working_set_gb': 7.99, 'private_gb': 1.69, 'system_free_gb': 9.86}.

## Test candidates and inference

- Test S1 1,732,544; pool S2 4,887,273 / S3 5,082,316; candidate pairs 126,274,337; candidates/S1 mean 72.9, median 42, p95 188, p99 255, max 533; S1 with zero candidates 0. Test-scale artifacts reused: True.
- Inference: 126,274,337 pairs scored in 766.5s; memory {'peak_working_set_gb': 7.99, 'private_gb': 1.47, 'system_free_gb': 11.94}.

## Final decisions (no labels — counts only)

| | validation (Step 3 model) | test, Step 3 model (diagnostic) | **test, final model** |
|---|---:|---:|---:|
| predicted pairs before rule | 1,465,359 | 6,311,401 | 5,523,440 |
| removed by one-to-one rule | 2,138 | 57,104 | 19,970 |
| final predicted pairs | 1,463,221 | 6,254,297 | 5,503,470 |
| S1 with zero predictions | 25,312 | 63,811 | 110,945 |
| S1 with 2+ predictions | 383,826 | 1,545,794 | 1,454,525 |
| predicted S2 | 705,805 | 3,040,162 | 2,652,224 |
| predicted S3 | 757,416 | 3,214,135 | 2,851,246 |
| predictions per S1 | 3.315 | 3.610 | 3.177 |
| zero-prediction rate | 5.73% | 3.68% | 6.40% |

By country (final): US: 3.207 predictions/S1, zero-prediction rate 6.13% (n=663,106); India: 3.042 predictions/S1, zero-prediction rate 7.20% (n=809,986); France: 3.518 predictions/S1, zero-prediction rate 4.62% (n=259,452)

Labelled stress reference (Step 3 model, validation S1 × full training pool): 1,725,775 predictions for 441,364 S1; macro F0.5 0.8400 (driven by orphaned targets, see experiments/TEST_SCALE_REPORT.md).

## Outputs and validation

- Submission: `C:\Users\kashish\Desktop\Amazon-ML-Challenge-2026\output\matching_results.tsv` (93,375,709 bytes, SHA-256 `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`).
- Candidate set scored by the matcher: `C:\Users\kashish\Desktop\Amazon-ML-Challenge-2026\experiments\final\candidate_pairs.tsv` (1,649,916,409 bytes) — needed as output/candidate_pairs.tsv in the final package.
- Official validator (matching_with_check_ids): exit 0 — `PASS — no blocking issues found. Safe to submit.`
- candidate_pairs.tsv streaming check (validator rules + matched ⊆ candidates): {'header': True, 'no_tab_rows': 0, 'dup_rows': 0, 'dup_ids_in_list': 0, 'bad_prefix': 0, 'unknown_ids': 0, 'missing_s1': 0, 'extra_s1': 0, 'matched_not_in_candidates': 0, 'rows': 1732544, 'ok': True}

Timings (s): {'check': 0.0, 'train_candidates': 2614.8, 'train_idf': 17.3, 'train_cos': 786.7, 'train_tc': 133.3, 'train_rows': 514.8, 'train_model': 1076.8, 'test_inputs': 2.9, 'test_infer': 767.2, 'decide': 4.5, 'write': 109.5, 'validate': 463.0}