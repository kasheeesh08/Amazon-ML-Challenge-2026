# Train-vs-test distribution audit (diagnostic only)

**Scripts and outputs:**
- `test_distribution_audit.py` → `results.json`, `test_matched_subset_s1_ids.txt`, `run.log`
- Two small read-only calibration passes: `singleton_calibration.json`/`.log` and `lookalike_singletons.json`/`.log`

**Scope:** saved artifacts only. Nothing was retrained or re-inferred, there is no new submission, and the threshold was not tuned. Test labels do not exist.

## Artifacts used

- **Training and validation:** stores, folds and ground truth (`experiments/blocking/features`, `experiments/validation_split`, `experiments/_eda_cache`).
- **Labelled environments:**
  - V, normal validation: `experiments/matching/cand_val.npy`, `X_val.npy`.
  - T, test-ratio: `experiments/diagnostics/trank/cand_T.npy`, `X_T.npy`, `poolT_s*.npy`.
  - S, full-pool stress: `experiments/test_scale/stress/cand_stress.npy`, `trank/X_S.npy`.
  - Step 3-model probabilities `trank/p_{V,T,S}_A.npy`.
- **Real test:**
  - `experiments/test_scale/cand_test.npy`, `features_test/`, `prob_test.npy` (Step 3 model, which allows like-for-like comparison with labelled data).
  - `experiments/final/test_prob.npy`, `test_pred_codes.npy` (the submitted final model; it cannot be evaluated on labelled data because it was trained on them).
- **Earlier results, read only:** `experiments/diagnostics/blocking_gap/results.json`, `france_error_analysis/results.json`, and the LOCO, France-threshold and target-context-ablation reports.

## 1. S1 country distribution

| Country | Train (all) | Fit | Validation | Test | Train share | Test share |
|---|---:|---:|---:|---:|---:|---:|
| US | 1,323,633 | 1,058,905 | 264,728 | 663,106 | 60.0% | **38.3%** |
| India | 883,188 | 706,552 | 176,636 | 809,986 | 40.0% | **46.8%** |
| France | 0 | 0 | 0 | 259,452 | 0% | **15.0%** |

Test shifts toward India, the weaker labelled country (T: India 0.9394 vs US 0.9643), and adds unlabelled France. The country-mix effect is already quantified at −0.0038 (blocking-gap report).

## 2. S2/S3 pool composition

| Country | Pool per S1: train / validation / test | S2 share of pool: train / test | Pool address empty: train / test | Share of train pool that are true targets |
|---|---:|---:|---:|---:|
| US | 4.674 / 4.674 / **5.756** | 48.8% / 49.0% | 3.59% / 2.89% | 74.0% |
| India | 4.680 / 4.680 / **5.824** | 48.8% / 49.0% | 2.97% / 2.37% | 74.0% |
| France | — / — / **5.531** | — / 49.0% | — / 3.00% | — |

- The test pool is **+23–24% larger per S1** in every country.
- Source composition and missing-address rates are **not** shifted; test addresses are slightly *more* complete.
- Pool non-Latin rates are unchanged (India 18.4% vs 18.2%).
- France has almost no postal codes (0.27%) and 1.74% initialism names, against 0.10–0.20% elsewhere (France error analysis).

## 3. Candidate structure (candidates per S1)

| Env | Mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| V-US / V-India | 64.2 / 50.0 | 39 / 39 | 141 / 76 | 173 / 117 | 214 / 197 |
| T-US / T-India | 65.5 / 51.1 | 39 / 40 | 146 / 80 | 179 / 118 | 222 / 187 |
| TEST US | 67.1 | 40 | 145 | 175 | 225 |
| TEST India | 77.0 | 47 | 157 | 192 | 255 |
| TEST France | 74.6 | 43 | 173 | 215 | **292** |

- Test India has more candidates than validation India (77.0 vs 50.0; p90 157 vs 76).
- France has the heaviest tail: 0.84% of France S1 have 300+ candidates.
- Generic names get about 2.5–3× more candidates: France 154 vs 50 (blocking-gap report).
- Test blocking behaves like the full-pool environment S. Its estimated cost is −0.0063 (blocking-gap report).

## 4–5. Score and prediction distributions

Step 3 model (comparable across labelled and test):

| | V-US | T-US | S-US | TEST-US | V-India | T-India | S-India | TEST-India | TEST-France |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predictions per S1 | 3.353 | 3.381 | 3.830 | **3.618** | 3.259 | 3.306 | 4.029 | **3.533** | **3.829** |
| S1 with no prediction | 5.58% | 5.37% | 2.81% | **3.52%** | 5.96% | 5.63% | 2.70% | **4.14%** | **2.68%** |
| **S1 whose best candidate has p < 0.10** | 4.07% | 3.50% | 0.51% | **0.77%** | 3.57% | 2.89% | 0.17% | **0.64%** | **0.63%** |
| S1 whose best candidate has p in [0.10, 0.69) | 1.51% | 1.85% | — | 2.74% | 2.34% | 2.68% | — | 3.37% | 1.82% |
| Predictions with p in [0.69, 0.80) | 2.49% | 2.69% | 5.71% | **5.52%** | 3.15% | 3.45% | 7.61% | **6.18%** | 4.36% |
| Precision in that band (labelled) | 74.8% | 66.2% | 18.9% | ? | 74.9% | 65.3% | 18.0% | ? | ? |
| Predictions with p ≥ 0.95 | 90.9% | 90.4% | 82.4% | 85.1% | 85.8% | 85.1% | 75.3% | 79.3% | 87.6% |
| Share of pool records predicted | 71.7% | 58.8% | — | 62.9% | 69.6% | 57.4% | — | 60.7% | **69.2%** |
| True singleton rate (labelled) | 5.58% | 5.58% | 5.58% | ? | 5.59% | 5.59% | 5.59% | ? | ? |

**Predicted matches per S1** (0 / 1 / 2 / 3 / 4 / 5 / 6+):
- T-US: 5.4 / 6.7 / 17.7 / 24.3 / 21.4 / 14.0 / 10.4%
- TEST-US: 3.5 / 6.4 / 15.4 / 23.2 / 22.4 / 15.8 / 13.3%
- TEST-France: 2.7 / 5.7 / 13.8 / 21.8 / 22.7 / 16.8 / 16.5%

On test, the mass moves from 0–2 predictions to 4–6+, not to 1.

**Submitted final model on test** (more conservative than the Step 3 model):
- Predictions per S1: US 3.207, India 3.042, France 3.518.
- S1 with no prediction: 6.13%, 7.20%, 4.62%.
- Best candidate below 0.10: 1.67%, 1.89%, 1.29%.
- Predictions in [0.69, 0.80): 3.14%, 5.75%, 2.59%.
- Pool records predicted: 55.7%, 52.2%, 63.6%.

## 6. Populations in test that differ from every labelled environment

1. **Almost every test S1 has a strong-looking candidate.** This is the largest shift found, and it doesn't depend on the model.
   - By model probability: S1 whose best candidate has p < 0.10 are 0.6–0.8% on test against 2.9–4.1% in V and T.
   - By name+address cosine (blocking-gap report): S1 with best cosine < 0.5 are **0.07–0.31%** on test, against 1.8–3.2% in V and T and 0.25–0.45% in S.
   - In labelled data, such "isolated" S1 are almost all singletons: 48–71% of singletons in V/T have best probability < 0.10, against 0.1–0.2% of non-singletons.
   - So the real test has either **far fewer singletons**, or **singletons that have near-duplicate lookalikes**.
2. **More predictions per S1, concentrated in the uncertain band.** Step 3 predictions per S1 are +7% on test (US and India). The share of predictions in [0.69, 0.80) roughly doubles (US 5.52% vs 2.69%).
   - On these prediction-side indicators, the test sits 30–90% of the way from T toward S.
   - On the target-context features it looks like T (PSI ≤ 0.07; ablation report).
   - **No labelled environment has both properties.**
3. **France predicts the largest share of its pool:** 69.2% (Step 3) / 63.6% (final), against about 61–63% (Step 3) for US and India. It also has the fewest S1 with no prediction.
4. **Country mix** (India-heavier; France new) and the **larger pool per S1** (+23%). Both were already quantified in earlier diagnostics.
5. **Not shifted:** source composition (S2 share 48.8% → 49.0%), missing addresses (slightly lower on test) and non-Latin rates.

## 7. Labelled subsets that resemble the test

**Test-matched subset** (`test_matched_subset_s1_ids.txt`): 242,212 validation S1, selected deterministically so that the cells *country (US/India) × Step 3 predicted-count bucket (0…6+) in T* have the non-France test shares.

| Existing baseline predictions evaluated on | All validation S1 | Country-mix reweighted | Country × predicted-count reweighted | **Test-matched subset** |
|---|---:|---:|---:|---:|
| V | 0.9628 | 0.9597 | 0.9618 | 0.9617 |
| T | 0.9543 | 0.9506 | 0.9524 | **0.9523** |
| T + test-sized blocking losses | 0.9480 | 0.9442 | 0.9461 | **0.9459** |

- The subset has 4.10% true singletons and 3.67 true matches per S1, against 5.58% and 3.46 overall.
- Matching on predicted counts therefore implicitly assumes the first reading above: fewer singletons and more matches per S1 in test.

**Lookalike-singleton calibration** (`lookalike_singletons.json`), singleton F0.5 by whether the singleton has a lookalike candidate:

| Env | All singletons | Best cosine ≥ 0.5 | Best cosine ≥ 0.7 | Best cosine < 0.5 |
|---|---:|---:|---:|---:|
| T-US | 0.894 | 0.854 | 0.795 | 0.984 |
| T-India | 0.854 | 0.793 | 0.686 | 0.956 |
| S-US | 0.454 | 0.441 | 0.395 | 0.813 |

- Under the second reading (singleton rate as in train, 5.58%, but every singleton has a lookalike), a T-like environment costs 0.0558 × (0.894 − 0.854 … 0.795) = **0.002–0.006 for US** and **0.003–0.009 for India**.
- The S-level singleton collapse (about 0.44, a loss of about 0.025) needs orphan-driven target context, which the real test does not show.

## 8. Could any shift explain about 0.03?

**Scenario ledger, non-France part** (arithmetic on measured values; not a prediction):

| Step | Estimate |
|---|---:|
| Local validation | 0.9628 |
| Test ratio / target context (T) | 0.9543 |
| Test-sized blocking | 0.9480 |
| Country mix | 0.9442 |
| Test-matched subset (first reading) | 0.9459 |
| Lookalike singletons (second reading, T-conditional) | about 0.936–0.942 |

**France** would then need about 0.80–0.87 to produce 0.924.
- The lower end is below the worst LOCO cross-country case (−0.075 → about 0.87).
- The upper end is reachable.

**Verdict:**
- **No single measured shift explains about 0.03 on its own.**
- The largest newly found shift (almost no isolated S1 in test) is real and model-independent. In labelled analogues, though, it is worth only a few thousandths unless test singletons are also orphan-like, and the measured target-context distributions argue against that.
- Together with the earlier effects (test ratio −0.0085, blocking −0.0063, country mix −0.0038, lookalike singletons −0.002 to −0.009), plus a France penalty in the range LOCO measured, the gap is **roughly accounted for at the pessimistic end** of each range. It is not proven at any point.

## 9. Comparison with previous diagnostics

| Diagnostic | Finding | Consistency with this audit |
|---|---|---|
| `t_rank` / target-context ablation | Target-context features look like T on test, not S | Confirmed. The S-like signals here are on the *prediction/singleton* side, not target context |
| LOCO | Unseen country costs 0.011–0.078 | France remains the largest unmeasurable term |
| France threshold | Weak lever (\|Δ\| < 0.001) | Consistent. The France-specific uncertain band is small (final model 2.59%) |
| Blocking gap | −0.0063 at test-sized pool; test blocking is S-like | Consistent (candidate structure) |
| France error analysis | France mix clean; fewer empty S1 is test-wide | Confirmed, and extended: fewer empty S1 goes with almost no isolated S1 in *every* country |
| Orphan-aware training | Real robustness under orphans; about +0.001 at the test ratio | Consistent. Orphan-driven singleton collapse is not what test shows |

## 10. What remains unexplained, and whether a change is justified

- **Unexplained:**
  - France's actual performance (no labels).
  - Whether the extra test predictions (+7% per S1, concentrated in [0.69, 0.80)) are mostly correct (more duplicates per business) or mostly wrong (lookalike false merges).
  - Depending on these, a residual of about 0 to 0.015.
- **Blocking modification:** not justified. It is bounded at about 0.006 and blocking is healthy.
- **Model modification:** not justified by labelled evidence. Every variant tested so far moves test-ratio by at most about 0.002.
- **The only open question an action could resolve** is which reading of the extra uncertain-band predictions is true, and only the leaderboard can answer it:
  - A single controlled probe that re-thresholds the *existing* final probabilities (e.g. 0.75–0.80; new file, current submission untouched) would discriminate the two readings.
  - In labelled T it costs little: the Step 3 model at 0.80 gives +0.0019 US / +0.0007 India on T, and −0.001 / −0.003 on V.
  - This requires your explicit approval, and its effect may be small at 3 decimals.
  - Otherwise, **keep the current submission**.

## Integrity

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/model_final.npz`, `camelcase_submission.zip` and `blocking_config.json` are unchanged (`results.json` → `hashes_before` / `hashes_after`, `unchanged: true`).
- Files created, all in `experiments/diagnostics/test_distribution_audit/`:
  - `test_distribution_audit.py`
  - `results.json`
  - `test_matched_subset_s1_ids.txt`
  - `singleton_calibration.json` and `singleton_calibration.log`
  - `lookalike_singletons.json` and `lookalike_singletons.log`
  - `run.log`
  - `TEST_DISTRIBUTION_AUDIT_REPORT.md`
- Cost: about 76 s for the main audit plus about 3 minutes for the calibration passes. No training or inference.
