# Blocking-miss feasibility analysis (read-only)

Script: `experiments/diagnostics/blocking_miss_analysis/blocking_miss_analysis.py`. Results are in `results.json`, the log in `run.log`. Runtime was 575 s.

**Scope:**
- Nothing in production was changed, and no full candidate set was regenerated. There was no retraining, no inference and no upload.
- The missed true pairs come from the **saved** candidate sets.
- Retrieval variants were re-run **only** for:
  - the S1 that have misses (22,866 in V, 53,248 in S)
  - a seeded random sample of 20,000 validation S1, for the candidate-cost estimate
- **Environments:**
  - V: validation pool.
  - S: the full training pool, **test-sized**. Earlier diagnostics showed the real test's blocking behaves like S.

## 1. Recall and misses

| | V | **S (test-sized pool)** |
|---|---:|---:|
| Candidate recall | 98.214% | **95.631%** |
| True pairs missed | 27,277 | **66,750** |
| S2 / S3 | 13,676 / 13,601 | 32,972 / 33,778 |
| US / India | 10,544 / 16,733 | 32,366 / 34,384 |
| S1 with at least one miss | 22,866 | 53,248 |
| Current candidates per S1 | 58.5 | 72.8 |

## 2. What the missed pairs look like

| Share of misses | V | S |
|---|---:|---:|
| Mean name cosine / mean address cosine | 0.26 / 0.29 | 0.38 / 0.37 |
| No shared name token | 48.8% | 35.8% |
| Name cosine ≥ 0.8 | 2.3% | 8.6% |
| Address cosine < 0.3 | 46.9% | 35.3% |
| Country equal | 100% | 100% |
| Postal code present on both sides / equal | 0.07% / 0.04% | 0.10% / 0.04% |
| House number present on both sides / equal / differs | 66.8% / 20.0% / 46.8% | 65.7% / 21.1% / 44.7% |
| Target name non-Latin | 43.5% | 31.9% |
| Target address empty | 24.6% | 21.6% |
| Target name domain-like | 5.6% | 9.8% |
| Exact name key equal (`name_n2` / `name_n3`) | 0% / 0% | 2.4% / 2.7% |
| First name token equal | 35.9% | 42.2% |
| Address digits equal | 11.4% | 13.5% |

## 3. Which blocker fails

- **Exact keys:**
  - In **V**, the only missed pairs whose key is equal are 2,369 on `house+addr_digits`. All 2,369 were dropped by the block cap (> 100 pool records).
  - In **S**, the misses with an equal key were **all** dropped by the cap:
    - `house+addr_digits`: 7,122
    - `name_n3`: 1,800
    - `name_n2` / `name_n2+country`: 1,611
    - `name_n1`: 335
    - `name_tok1+house`: 39
    - `name_tok1+addr_digits`: 13
  - These overlap.
- **Retrieval:** no missed pair ranks in the top 10 of the production name+address or address channels (sanity check: 0). The misses fall below the top 10, or outside what the 20,000-postings budget scans.

## 4–5. Failure modes (mutually exclusive, in priority order) and recovery by candidate rules

| Failure mode | V misses (share) | **S misses (share)** | S recovered by: name+address K=20 | address K=20 | name-only K=10 (new) | **name+address budget 200k** |
|---|---:|---:|---:|---:|---:|---:|
| Target name non-Latin (script differs) | 11,868 (43.5%) | 21,278 (31.9%) | 371 | **2,547** | 0 | 34 |
| House number differs | 5,032 (18.4%) | 18,316 (27.4%) | 3,355 | 767 | 519 | **8,923** |
| Target address empty | 6,588 (24.2%) | 14,310 (21.4%) | 1,819 | 0 | 828 | **5,230** |
| Other partial name/address differences | 1,469 (5.4%) | 8,584 (12.9%) | 1,568 | 983 | 142 | **4,693** |
| Name shares no token (noisy/domain/different) | 2,016 (7.4%) | 3,566 (5.3%) | 128 | 389 | 0 | 16 |
| Address dissimilar (address cosine < 0.3) | 304 (1.1%) | 696 (1.0%) | 75 | 1 | 30 | 236 |
| **Total recovered (S)** | | | **7,316** | 4,687 | 1,519 | **19,132** |
| **Total recovered (V)** | | | 3,650 | 3,355 | 172 | 4,626 |
| New candidates per S1 (S / V; random-sample estimate) | | | 16.6 / 17.0 | 17.5 / 17.0 | 10.8 / 7.5 | **6.7 / 5.8** |

K = 50 recovers 15,162 in S (V: 7,306), at about +70 candidates per S1.

**Cap raise on exact keys:**
- It could recover at most about 8–9k of the S misses; most are `house+addr_digits`, and they overlap with retrieval.
- Blocks larger than 100 records mean about 20–100× more pairs per affected S1, so it is **not efficient**.

**Non-Latin misses:**
- These are the largest mode in V and still 32% in S.
- No token-level blocker bridges scripts. Only address K=20 helps (+2,547).
- Transliteration without external resources is out of scope.

## 6. The one promising rule

**Add a second name+address TF-IDF retrieval channel: top-10 per source with `max_postings` = 200,000**, alongside the existing 20,000-budget channel.
- **Recovery:**
  - **19,132 of 66,750 S misses (28.7%)**, which raises S candidate recall from 95.63% to about **96.88%**.
  - 4,626 of 27,277 V misses (17.0%).
  - It recovers the house-number-differs, empty-address and partial-difference modes, which are exactly the modes that grow with pool size.
- **Why it works:** in a test-sized pool, common tokens have long posting lists. The 20,000-posting budget (rarest tokens first) stops before scanning the tokens the true match shares.
- **Why add a channel rather than change the existing one:**
  - Replacing the budget *swaps* candidates.
  - The Step 2 design measured the channel's standalone recall@10 **falling** slightly with larger budgets (91.67% at 20k, 91.62% at 100k, 90.02% at 500k on the fit fold), so a replacement can lose pairs.
  - An additional channel can only add candidates.
- **Candidate cost:** about **+6.7 candidates per S1** in S (+9%; about +11.6M pairs on test) and +5.8 per S1 in V. That is by far the best recovery per candidate.
- **Implementation effort:** uses the existing `retrieval_channel` with one parameter changed, and adds one entry to the blocking config. The retrieval itself is about 6× slower per query: about 200 s for 73k S1 on the full pool, against about 35 s at the 20k budget.

**Estimated F0.5 value (scenario, not a measurement):**
- The blocking-gap hybrid put the whole S-vs-T blocking loss at about −0.0063. Recovering about 29% of S misses, which are harder pairs with lower matcher recall and precision, suggests roughly **+0.001 to +0.003** F0.5.
- Any gain requires new candidates to be featurised and scored. The current model would need to be applied to the enlarged candidate set, and ideally the model would be retrained with the same blocking.

## Recommendation

- **Worth testing only as a scoped, measured step.** It is the largest single lever found in this project (larger than any threshold change), but still modest.
- **Cheapest next experiment** (not run):
  - In environment S, generate only the additional 200k-budget channel for all 441k validation S1.
  - Measure the union candidate recall and candidate count exactly.
  - Score the new candidates with the **existing** Step 3 model: features plus the target context recomputed on the enlarged set.
  - Compare macro F0.5 against 0.84002 (S), and against the T/hybrid estimate.
- **Estimated compute:** roughly 30–60 minutes. No retraining or test inference.
- Only if the end-to-end gain is at least about +0.002 would a production rebuild be justified: test candidates, test features and inference, about 2–3 h, plus ideally retraining.
- **If compute or credits are tight, the rational choice is to stop here and keep the current submission.**

## Integrity

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `camelcase_submission.zip`: `93b8db37fd2c8a79d74cc80ceb8051451b546f817393b5b1278f76426ed374a8`, unchanged.
- `experiments/final/model_final.npz`: `1bac911596c3d42eca3dc0c0c01660af3eb4bd6fe27f5e68299f3f4c02866755`, unchanged. All 168 files in `experiments/final/` and all 34 protected files are unchanged.
- Files created, all in `experiments/diagnostics/blocking_miss_analysis/`:
  - `blocking_miss_analysis.py`
  - `results.json`
  - `run.log`
  - `BLOCKING_MISS_ANALYSIS_REPORT.md`
