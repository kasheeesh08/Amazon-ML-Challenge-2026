# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** camelcase  
**Team Members:** Kashish, Satvik Akshintala, Behlah Kathavala, Aakanksha Desai  
**Submission Date:** leaderboard predictions submitted 26 September 2026; final package 2 October 2026

---

## 1. Executive Summary
We use a two-stage pipeline: high-recall candidate generation (blocking), then a pairwise matcher.
- **Blocking:** TF-IDF top-K retrieval, run separately for S2 and S3 within each country, combined with capped
  exact-key joins.
- **Matcher:** a from-scratch histogram gradient-boosted-trees model over 50 similarity and candidate-context
  features.
- **Decision rule:** a fixed threshold (0.69) plus a one-S1-per-target rule.

On a leakage-safe held-out validation split, blocking kept **98.2144%** of true pairs, and the full pipeline scored
an official macro **F0.5 of 0.962847**. The main innovation is the candidate-context features: how a candidate
ranks among its S1's candidates, and how its S1 ranks among all S1 competing for the same target record.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from a streaming EDA over the full data (train: 2.21M S1, 5.03M S2, 5.29M S3):
- **Ground truth:**
  - 5.58% of S1 are singletons.
  - Each S1 has 3.46 matches on average, and at most 11.
  - S2 and S3 contain internal duplicates.
  - **No S2/S3 record matches more than one S1.**
  - There are no cross-country matches.
- **Field quality:**
  - S1 is clean. S2 and S3 carry the noise: casing, legal-form changes, typos, word order, reordered and
    abbreviated addresses, `NC` vs `North Carolina`, leading-zero house numbers, landmarks, domain-style names,
    junk prefixes, and 5–9% native-script (Indic) names.
  - About 3.3% of S2/S3 addresses are empty.
  - Postal codes appear in fewer than 3% of records, so PIN/ZIP blocking isn't viable.
- **Test differences:** test adds an unseen country (France, 15% of test), and test pools are larger per S1
  (5.75 records against 4.68).

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (with candidate-set context features and one-to-one post-processing)  
**Core Innovation:** Candidate-context features (rank / score gap / competitor count of a pair within its S1's candidates and among all S1 retrieving the same target), combined with an exact numba TF-IDF top-K retriever that bounds work per query.

---

## 3. Candidate Generation (Blocking)
We reduced the comparison space in three ways:
- **TF-IDF retrieval:** cosine similarity with binary term frequency and smooth IDF from the pool. It searches S2
  and S3 separately and within the query's country (training has 0 cross-country matches). The rarest query tokens
  are scanned first, up to 20,000 postings.
- **Retrieval channels:** name+address tokens (words, character trigrams of the joined name, address words, digit
  groups, house number), top 10 per source; and address tokens, top 10 per source.
- **16 exact-key joins**, run per source, with cap 100: a block with more than 100 pool records is dropped entirely.
  The keys are:
  - `name_n1`, `name_n2`, `name_n3`
  - `addr_n1`, `addr_n2`
  - `name_n2+country`, `addr_n2+country`
  - `name_n1+addr_n1`, `name_n2+addr_n2`, `name_n3+addr_n2`
  - `name_n3+house`, `name_tok1+house`, `name_tok1+addr_digits`
  - `postal+name_tok1`, `postal+house`
  - `house+addr_digits`

  Key meanings:
  - `name_n1`: normalised name
  - `name_n2`: legal forms removed, initials joined
  - `name_n3`: `name_n2` tokens sorted
  - `name_tok1`: first `name_n2` token
  - `addr_n1`: normalised address
  - `addr_n2`: sorted set of expanded address tokens
  - `addr_digits`: sorted set of digit groups

  Exact joins search the whole pool of a source. They are restricted to the same country only when the key contains
  `country`.
- **Country handling:** retrieval is country-partitioned. A query whose country is empty or absent from the pool
  falls back to the whole pool of that source. France is present in the test pools, so it is retrieved within France.
- **Combination:**
  - The candidate set is the deduplicated union of all 18 channel outputs, for S2 and S3.
  - Nothing is ranked, filtered or capped after the union, and no candidates are hard-coded.
  - The union is exactly the set the GBDT scores, and exactly `output/candidate_pairs.tsv`.
- **Evaluated but NOT used in production:**
  - a `name`-only retrieval channel (71.3% recall at K=10)
  - K=50 retrieval
  - a second `nameaddr` channel with a 200k-postings budget (analysed after the leaderboard submission, never
    applied)

The rest of this section covers the blocking-specific details:
- **Blocking keys used:** normalised/sorted name keys, normalised address keys, house number, address digit set, postal code (rare), first name token; TF-IDF over name word + character-trigram tokens and address tokens.
- **Candidate pairs generated:**
  - Validation: 25,826,498 pairs (58.5 per S1, reduction ratio 0.99997).
  - Test: 126,274,337 pairs for 1,732,544 S1 (72.9 per S1, median 42, maximum 533; no S1 without candidates).
- **How you ensured true matches were not lost:**
  - All budgets, K values and the union were chosen on a test-sized training pool with pre-stated rules, then
    measured on a held-out validation split.
  - Candidate recall is **98.2144%** on validation (100% for casing/punctuation, legal-suffix, word-order and
    address-reordering variants).
  - Candidate recall is **95.63%** when validation S1 are matched against the full 10.3M training pool.
  - The remaining misses are mostly native-script names, empty addresses and changed house numbers.

---

## 4. Matching Model

**Features used** (50 per pair):
- Name features: equality of five normalised name keys; set size, intersection, Jaccard, containment (both directions), IDF-cosine and IDF-weighted Jaccard over name word + character-trigram tokens.
- Address features: equality of address / house-number / digit-set / postal keys and their presence; the same eight set similarities over address tokens; name+address IDF cosine.
- Other: source (S2 vs S3); target/query flags (empty address, non-Latin name, domain-like name, junk prefix, postal present, landmark); candidate context — rank, score gap and group size within the S1's candidates (per source and overall) and among all S1 that retrieved the same target (`t_rank`, `t_gap`, `t_ncand`).

**Model type:**
- A histogram gradient-boosted decision tree ensemble, implemented from scratch in numba.
- Binary log-loss; 300 trees, depth 6, learning rate 0.1, L2 λ = 1, minimum child hessian 20, 64 quantile bins; no
  subsampling, deterministic.
- Baseline: a weighted L2 logistic regression.

**Threshold selection method:**
- The model, tree count, threshold and one-to-one rule were chosen by maximising the official macro F0.5 on an
  internal split of the training fold, never on the validation labels.
- The selection was threshold 0.69 with the one-S1-per-target rule on.
- The validation-optimal threshold would have been 0.70, confirming the choice transfers.

**Training procedure:**
- All positive candidate pairs are kept, plus a deterministic 10% hash sample of negatives with weight 10.
- The submitted model uses all training S1 blocked against the full training pool: 160.55M candidate pairs, giving
  22.63M training rows (7.31M positives).

---

## 5. Results & Error Analysis

**Validation methodology:**
- An 80/20 split at the S1-entity level, stratified by country, match count and S2/S3 pattern.
- True targets follow their S1; distractors are split proportionally. No S1 or S2/S3 record is shared between the
  folds.
- The metric is the official per-entity macro F0.5, implemented exactly and cross-checked against the official
  validator's format rules.

- **F0.5 Score (macro):**
  - Validation: **0.962847** on 441,364 S1, against 0.944584 for the logistic-regression baseline. Pair precision
    is 98.22%, pair recall 94.08%, and singleton F0.5 91.93%.
  - **Public leaderboard: ≈ 0.924** for the submitted `output/matching_results.tsv` (threshold 0.69). The team
    recorded this value to three decimals; more decimals were not kept.
  - A later leaderboard probe re-thresholded the same probabilities at 0.72 and scored 0.923607. It is not part of
    this submission, and it confirmed that the threshold was not a lever.
- **Common false positives (wrong merges):** near-identical records at the same address that differ only in legal form or an extra word ("Corner Tattoo" vs "Corner Tattoo Co"); same name at a different address; same address with a different name; mismatched house numbers.
- **Common false negatives (missed matches):** native-script names without transliteration (e.g. Malayalam/Devanagari vs Latin), targets with an empty address, changed house numbers, heavily typo'd names.

**Test-scale stress test** (validation S1 against the full 10.3M training pool):
- Macro F0.5 was 0.8400. 90% of the extra false positives came from "orphaned" targets whose owner S1 wasn't
  queried, which inflates the `t_rank` signal.
- Restricted to validation-pool records, the same predictions score 0.9560.
- The real test set queries every S1, and its target-context feature distributions matched validation (PSI ≤ 0.07).

---

## 6. Conclusion
A carefully validated blocking stage (98.2% candidate recall) plus a from-scratch gradient-boosted matcher reaches
0.9628 macro F0.5 on held-out validation.

Key lessons:
- Precision-weighted F0.5 rewards a conservative threshold and a one-to-one target rule.
- Candidate-context features are powerful, but they depend on every target's owner being present among the queries.
- France (unseen in training) and native-script names remain the main open risks.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run_pipeline.py`: the single entry point (data → blocking → matching → output), with checkpointed phases.
- `src/ber/normalize.py`: normalisation, exact keys, tokens, flags.
- `src/ber/features.py`: chunked multi-process featurisation into on-disk stores.
- `src/ber/blocking.py`: exact-key joins and the numba TF-IDF top-K retriever.
- `src/ber/match_features.py`: the 50 pair features.
- `src/ber/gbdt.py`: the histogram GBDT.
- `src/ber/matcher.py`: the decision rule and TSV writer.
- `src/ber/pipeline.py`: chunked candidate union, partitioned target context, ground-truth loader.
- `src/config/blocking_config.json` and `src/config/matcher_config.json`: the fixed configuration.
- `README.md`: how to reproduce; `requirements.txt`: numpy, numba, llvmlite and scipy, pinned.

Reproduce with `python src/run_pipeline.py --train-dir <dataset>/train --test-dir <dataset>/test --work-dir work --output-dir output`.

**Provenance of the submitted files:**
- The submitted outputs were produced by the project's experiment driver.
- `run_pipeline.py` consolidates that driver around the byte-identical `ber` modules and configuration.
- Built from this zip in a clean venv (`pip install -r requirements.txt`), it reproduced the driver's
  `matching_results.tsv` and `candidate_pairs.tsv` **byte for byte** on a labelled smoke subset.
- The full-scale run (about 3.2 hours) was not repeated. The submitted files are the originals.

**Final-run SHA-256:**
- `matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`
- `candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`
- model: `1bac9115…66755`

### Compliance
- **Data:** only the provided training and test files are used. There is no external data, API, geocoding,
  business registry or internet augmentation.
- **Code:** it makes no network access of any kind.
- **Models:** no pretrained models or embeddings. The only model is the from-scratch GBDT trained on the provided
  labels: 300 trees of depth ≤ 6.
- **Dependencies:** numpy and scipy (BSD-3-Clause), numba and llvmlite (BSD-2-Clause).

### B. Additional Results

| blocking channel (validation; recall of the channel alone) | candidate recall |
|---|---:|
| name+address retrieval, **top 10 per source (used)** / top 50 (evaluated, not used) | 93.97% / 96.30% |
| address retrieval, **top 10 (used)** / top 50 (evaluated, not used) | 87.24% / 91.13% |
| name-only retrieval, top 10 (evaluated, not used) | 71.31% |
| exact name keys (cap 100, used) | 25.7–52.5% |
| union of the production channels | **98.2144%** |

**Validation F0.5 by country:** US 0.9712, India 0.9503.

**Test predictions** (unlabelled, counts only):
- 5,503,470 predicted matches for 1,732,544 test S1 (3.18 per S1).
- 110,945 S1 (6.40%) have no prediction.
- 2,652,224 matches come from S2 and 2,851,246 from S3.

**Limitations:**
- No labels exist for France.
- The final model is trained on all labelled data, so it has no separate hold-out.
- Sensitivity to orphaned targets.
- No transliteration of native-script names.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
