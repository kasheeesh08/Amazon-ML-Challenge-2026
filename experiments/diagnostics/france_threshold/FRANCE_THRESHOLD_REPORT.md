# France-only threshold sensitivity (diagnostic only)

Script: `experiments/diagnostics/france_threshold/france_threshold.py`. Raw numbers are in `results.json` and the log in `run.log`.

- **No France labels exist.** Nothing below is a measured France F0.5 or a leaderboard estimate.
- **The submission is unchanged.** Nothing was retrained or re-inferred, no submission file was written, and nothing was uploaded.

## Method

- **Baseline:** the exact submitted predictions, `experiments/final/test_pred_codes.npy`. Two checks passed:
  - The codes are pair-for-pair identical to `output/matching_results.tsv` (1,732,544 rows, 5,503,470 pairs).
  - They are identical to re-applying the final rule (probability ≥ 0.69 with the one-S1-per-target rule) to `experiments/final/test_prob.npy` over `experiments/test_scale/cand_test.npy`.
- **Simulation at threshold t:** every US and India prediction is kept unchanged (4,590,782 pairs). France predictions are kept only if the final-model probability is ≥ t.
  - This is a pure filter. The candidates, probabilities and the one-S1-per-target rule are unchanged. A subset of a valid one-per-target set stays valid, and this was re-checked.
- **Other inputs:**
  - France S1 come from the test S1 feature store (`experiments/test_scale/features_test`).
  - `t_rank` and `t_ncand` come from `experiments/test_scale/t_rank_test.npy` and `t_ncand_test.npy`.
  - House-number conflict, same name with a different address, and legal-form-only name difference come from the Step 3 feature code over the test IDF.
  - "Generic name" means the S1's normalised name (`name_n2`) is shared by at least 10 France S1.

## Results

| | baseline (0.69) | 0.80 | 0.82 | 0.85 | 0.88 |
|---|---:|---:|---:|---:|---:|
| Total predictions | 5,503,470 | 5,479,787 | 5,474,811 | 5,465,769 | 5,454,989 |
| France S1 | 259,452 | 259,452 | 259,452 | 259,452 | 259,452 |
| **France matches** | 912,688 | 889,005 | 884,029 | 874,987 | 864,207 |
| France matches per France S1 | 3.518 | 3.426 | 3.407 | 3.372 | 3.331 |
| **France S1 with no match** | 11,981 (4.62%) | 12,879 (4.96%) | 13,061 (5.03%) | 13,322 (5.13%) | 13,646 (5.26%) |
| **Predictions removed** | — | 23,683 | 28,659 | 37,701 | 48,481 |
| **% of France predictions removed** | — | 2.59% | 3.14% | 4.13% | 5.31% |
| France S1 losing ≥1 prediction | — | 22,252 | 26,632 | 34,281 | 43,013 |
| France S1 becoming empty | — | 898 | 1,080 | 1,341 | 1,665 |
| Removed: `t_rank` = 0 | — | 21,488 (90.7%) | 26,045 | 34,370 | 44,411 (91.6%) |
| Removed: `t_rank` = 0 on an uncontested target (`t_ncand` = 1) | — | 1,649 | 1,956 | 2,508 | 3,203 |
| Removed: mean `t_ncand` | — | 14.0 | 13.9 | 14.0 | 13.9 |
| Removed: house-number conflict | — | 7,472 (31.5%) | 8,391 | 9,701 | 10,971 (22.6%) |
| Removal rate among house-conflict / other predictions | — | 25.7% / 1.8% | 28.9% / 2.3% | 33.4% / 3.2% | 37.8% / 4.2% |
| Removed: same name, different address | — | 6,496 | 7,802 | 10,045 | 12,453 |
| Removed: target address empty | — | 5,975 | 7,143 | 9,121 | 11,175 |
| Removed: legal-form-only name difference | — | 6,746 | 7,503 | 8,505 | 9,400 |
| Removal rate: generic-name S1 / unique-name S1 | — | 2.94% / 2.60% | 3.57% / 3.14% | 4.67% / 4.14% | 6.02% / 5.31% |
| S1 with no match: generic / unique name | 4.37% / 4.65% | 4.73% / 4.98% | 4.81% / 5.05% | 4.95% / 5.15% | 5.09% / 5.26% |
| One-S1-per-target satisfied | yes | yes | yes | yes | yes |
| US + India predictions identical to baseline | yes | yes | yes | yes | yes |
| Removed targets that another S1 claimed at ≥ 0.69 (a full rule re-run could reassign them) | — | 168 (all rival claims from France S1) | 223 | 362 | 514 |

Probability distribution of removed predictions (counts; each band is contained in the next threshold):
- [0.69, 0.72): 6,218
- [0.72, 0.75): 6,236
- [0.75, 0.78): 6,617
- [0.78, 0.80): 4,612
- [0.80, 0.82): 4,976
- [0.82, 0.85): 9,042
- [0.85, 0.88): 10,780

`t_rank` distribution of removed predictions at 0.88:

| `t_rank` | 0 | 1 | 2 | 3–4 | 5–9 | 10+ |
|---|---:|---:|---:|---:|---:|---:|
| Removed | 44,411 | 1,993 | 612 | 456 | 398 | 611 |

Other lower thresholds scale proportionally; see `results.json`.

**Affected record IDs:**
- `removed_france_pairs_upto_0.88.tsv` lists all 48,481 pairs removed at 0.88. For each pair it gives the S1 ID, the target ID, the probability, the smallest threshold that removes it, `t_rank`, `t_ncand`, the diagnostic flags, and the name count.
- The first 15 S1 that become empty at each threshold are in `results.json`. At 0.80 the first five are S1-521482584, S1-325608211, S1-324095800, S1-679163581 and S1-385954406.

Examples of pairs removed at 0.80, from the test files:
- "Afps & Cie SA", 75 Boulevard de Pyla ↔ "Afps & Cie SAS", 76 Boulevard De Pyla. p = 0.72; house conflict.
- "Varois (France) Comite SAS", 13 Rue du Maréchal Kellermann ↔ "VAROIS (FRANCE) COMITE S.A.R.L.", 22 Rue du Maréchal Kellermann. p = 0.71.
- "Lille Maternelle", 25 Rue Nationale ↔ "Lille Maternelle", 65 R. de Cambrai. p = 0.75; the name is shared by 56 France S1.
- "Solidarite Cie", 271 Avenue Antoine de Saint Exupéry ↔ "SC", 271 Ave Antoine de Sasnt Exupéry. p = 0.78; the target name is an initialism.
- "Club des Libre", 4 Rue de Bacalan ↔ "Club des Libre", no address. p = 0.69.

## Labelled comparison (validation S1 only; NOT France)

The same filter was applied to existing validation predictions in environment V (normal) and T (test-like pool ratio). Each cell shows macro F0.5 for that country, and in brackets the share of the removed predictions that were true matches.

| Case | Env | Own threshold | 0.80 | 0.82 | 0.85 | 0.88 |
|---|---|---:|---:|---:|---:|---:|
| Current model, US (in-distribution) | V | 0.9712 | 0.9702 (74.8%) | 0.9694 | 0.9678 | 0.9654 (80.1%) |
| Current model, India (in-distribution) | V | 0.9503 | 0.9475 (74.9%) | 0.9460 | 0.9429 | 0.9383 (80.7%) |
| LOCO over-merge analogue: India-only model on US | V | 0.9607 | 0.9624 (64.5%) | 0.9625 | 0.9623 | 0.9618 (69.8%) |
| LOCO under-merge analogue: US-only model on India | V | 0.8735 | 0.8670 (83.1%) | 0.8642 | 0.8591 | 0.8530 (85.0%) |
| Current model, US | T | 0.9643 | 0.9662 (66.2%) | 0.9658 | 0.9651 | 0.9635 (72.6%) |
| Current model, India | T | 0.9394 | 0.9401 (65.3%) | 0.9393 | 0.9374 | 0.9339 (72.5%) |
| LOCO over-merge analogue: India-only model on US | T | 0.9500 | 0.9546 (54.1%) | 0.9553 | 0.9563 | 0.9564 (60.8%) |
| LOCO under-merge analogue: US-only model on India | T | 0.8647 | 0.8602 (75.0%) | 0.8582 | 0.8543 | 0.8492 (78.1%) |

- In every labelled case, **most removed predictions were true matches** (54–85%).
- Raising the threshold helps only where over-merging is strong:
  - the over-merge analogue: +0.0017 on V and +0.0046–0.0064 on T
  - the in-distribution models in the test-like environment T, at 0.80: +0.0019 US, +0.0007 India
- It hurts in normal validation (−0.001 to −0.012) and in the under-merge direction.
- Roughly, it pays off only when fewer than about 65–70% of the removed predictions are true. **That share is unknown for France.**

## Conclusions

**A. What changes at each threshold.**
- 0.80 removes 23,683 France predictions (2.59%) and empties 898 more France S1, raising France S1 with no match from 4.62% to 4.96%.
- 0.88 removes 48,481 (5.31%) and takes France S1 with no match to 5.26%.
- About 91% of the removed predictions are the best-claiming S1 for a contested target (`t_rank` = 0, mean `t_ncand` about 14).
- House-number conflicts are hit disproportionately: at 0.80, 25.7% of them are removed against 1.8% of other predictions. Same-name/different-address pairs and targets with an empty address are also over-represented.
- US and India are untouched, and the one-per-target rule holds at every threshold.

**B. Over-merging hypothesis.**
- Supports it:
  - France has the fewest S1 with no match (4.62%, against 6.13% US and 7.20% India).
  - France's generic-name S1 are left empty *less* often than unique-name S1. In the labelled countries, generic names are left empty *more* often; in India, 9.3% against 6.2%.
  - These match the LOCO over-merge signature.
- Contradicts or weakens it:
  - At 0.80 France loses 2.59% of its predictions, about the same as US in-distribution in V (2.5%) and less than India (3.2%). France is not unusually concentrated near the threshold.
  - 91% of France predictions are at probability ≥ 0.95.
  - The threshold barely changes generic-name behaviour (removal rate 2.94% generic against 2.60% unique), and generic-name S1 are still left empty less often than unique ones at every threshold.
  - Any confident over-merges on generic French names are therefore **not** what a threshold of 0.80–0.88 removes. Whether such confident errors exist is UNRESOLVED.

**C. Risks of a France-only threshold.**
- In every labelled analogue, 54–85% of the removed predictions were true matches. If France behaves like an in-distribution country, recall is lost and F0.5 goes down, as seen in V.
- House-conflict and address-empty pairs are removed disproportionately, but in the labelled reference these categories were still 83–95% precise.
- France's low empty-S1 rate could simply mean France has fewer true singletons; that is unknowable without labels.
- The rule depends on the country label (legitimate as an input feature, but it is a post-hoc rule fitted without France labels).

**D. Is one threshold worth testing on the leaderboard?** Only weakly.
- Scaled by France's share of test (15%), the labelled analogues imply an overall effect somewhere between about −0.0006 (in-distribution V behaviour) and about +0.001 (over-merge T behaviour). Both are below the 3-decimal resolution of the reported 0.924.
- The leaderboard's decimal precision and submission limits are UNRESOLVED.
- It is worth one submission only if submissions are cheap and the score shows at least 4 decimals. It is not a lever for the roughly 0.03 gap.

**E. If tested, which threshold?** **0.80**.
- It has the smallest downside in the labelled analogues: −0.001 US / −0.003 India in V, and it is the only threshold that improves both in-distribution T cases (+0.0019 / +0.0007).
- It captures about 72% of the over-merge analogue's T gain (+0.0046 of +0.0064).
- It removes the fewest France predictions (23,683) and empties the fewest S1 (898).
- Higher thresholds increase the downside for a small extra gain that appears only in the over-merge analogue.

## Integrity

- `output/matching_results.tsv`: SHA-256 `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, the same before and after.
- `output/candidate_pairs.tsv`: SHA-256 `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, the same before and after.
- All 34 protected production files (src, configs, models, `experiments/final`, `submission_package`, zip) are unchanged (`hashes_before.json` vs `results.json` → integrity).
- Files created, all in `experiments/diagnostics/france_threshold/`:
  - `france_threshold.py`
  - `hashes_before.json`
  - `results.json`
  - `removed_france_pairs_upto_0.88.tsv`
  - `run.log`
  - `FRANCE_THRESHOLD_REPORT.md`
