# Business Entity Resolution — Amazon ML Challenge 2026 (team camelcase)

Reproducible pipeline for the submitted solution. It regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the provided train/test data.

## 1. Problem Overview
Each Source 1 (S1) business record must be linked to every Source 2 (S2) and Source 3 (S3) record that describes the
same real-world business.
- S1 is clean.
- S2/S3 are noisy: casing, legal forms, typos, word order, abbreviated or reordered addresses, empty fields and
  native-script names.
- The score is the per-S1 F0.5 (precision-weighted), macro-averaged over all S1.

The solution has two stages:
1. **Blocking:** candidate generation with high recall.
2. **A pairwise matcher:** a gradient-boosted decision tree ensemble implemented from scratch. It is followed by a
   fixed decision rule.

## 2. Directory Structure
```
code/business_entity_resolution/
├── README.md                     this file
├── requirements.txt              pinned dependencies
└── src/
    ├── run_pipeline.py           single entry point (all phases, checkpointed)
    ├── config/
    │   ├── blocking_config.json  blocking configuration (16 exact-key joins + 2 TF-IDF retrieval channels)
    │   └── matcher_config.json   GBDT hyperparameters, threshold 0.69, one-to-one rule, negative sampling
    └── ber/
        ├── normalize.py          text normalisation, exact keys, blocking tokens, record flags
        ├── features.py           chunked multi-process featurisation of a source TSV into an on-disk store
        ├── blocking.py           exact-key joins and the numba TF-IDF top-K retriever
        ├── match_features.py     the 50 pairwise features
        ├── gbdt.py               histogram GBDT (numba)
        ├── matcher.py            decision rule (threshold + one-S1-per-target); also an unused LogReg baseline
        ├── ids_tsv.py            competition TSV header/format helpers
        └── pipeline.py           chunked candidate union, partitioned target context, ground-truth loader
```

## 3. Input Data
Place the official data as follows. Any location works, because the paths are passed on the command line.
```
<dataset>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<dataset>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
- **Sources** (TSV, UTF-8, with a header): `entity_id, business_name, business_address, country`.
- **`train_ground_truth.tsv`:** `source1_entity_id<TAB>matched_entity_ids`, with the IDs comma-separated. This is
  the only label source. It is used only to train the GBDT.
- **Test files:** featurised, blocked and scored. No test labels exist or are used.

## 4. Environment Setup
The submitted outputs were produced with **Python 3.11.9 (64-bit, Windows 11)**.
```
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` pins `numpy==2.4.6`, `numba==0.65.1`, `llvmlite==0.47.0` and `scipy==1.17.1`.
- `psutil` is optional. It is used only for memory logging and a low-memory guard.
- No pandas, sklearn, LightGBM, XGBoost or deep-learning framework is used.

## 5. Pipeline Execution
Run from this folder (`code/business_entity_resolution/`):
```
python src/run_pipeline.py --train-dir <dataset>/train --test-dir <dataset>/test --work-dir work --output-dir output
```
This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Each phase checkpoints to `--work-dir`, so rerunning the same command resumes an interrupted run.
- `--phase <name>` runs a single phase.
- `--workers N` sets the number of featurisation processes (default 6).

| order | phase | what it does |
|---|---|---|
| 1 | `featurise` | normalises all six source files into on-disk feature stores (keys, token CSR, flags) |
| 2 | `train_candidates` | blocks **all** training S1 against the **full** training S2/S3 pool (same config as test) |
| 3–5 | `train_idf`, `train_cos`, `train_tc` | builds pool IDF tables, the name+address cosine, and target-context features |
| 6 | `train_rows` | keeps all positive candidate pairs plus a deterministic 10% of negatives (weight 10), written to disk |
| 7 | `train_model` | trains the histogram GBDT (`config/matcher_config.json`) |
| 8 | `test_candidates` | blocks all test S1 against the full test S2/S3 pool |
| 9–11 | `test_idf`, `test_cos`, `test_tc` | same as phases 3–5, on the test candidates |
| 12 | `test_infer` | computes a GBDT probability for every test candidate pair |
| 13 | `decide` | applies threshold 0.69 and the one-S1-per-target rule |
| 14 | `write` | writes `matching_results.tsv` and `candidate_pairs.tsv` |

Data → normalisation (`featurise`) → blocking/candidate generation (`*_candidates`) → feature generation
(`*_idf/cos/tc`, `train_rows`) → model training (`train_model`) → inference (`test_infer`, `decide`) → output (`write`).

Optional format check with the official validator (from the challenge's `utils/`):
```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <dataset>/test
```

## 6. Blocking Strategy
Exact configuration: `src/config/blocking_config.json` (SHA-256 `91a38976…42e0f5`). It was chosen on an internal
fit fold of the training data and was never tuned on test.

**Normalisation** (`ber/normalize.py`):
- NFKC, casefold, Latin diacritics stripped, punctuation turned into spaces, `&` → `and`.
- Names:
  - runs of single letters are joined (`l l c` → `llc`);
  - legal-form tokens are removed (inc, llc, ltd, pvt, private, corp, co, company, plc, the, SARL, SAS, …).
- Addresses:
  - split on commas;
  - street abbreviations expanded (`rd` → `road`);
  - a state abbreviation is expanded when its component is the state alone or the state plus a 5/6-digit code
    (`NC` → `north carolina`);
  - digit groups have leading zeros removed, and a postal code and house number are extracted.

**Keys per record:**
- `name_raw`
- `name_n1`: normalised name
- `name_n2`: initials joined, legal forms removed
- `name_n3`: `name_n2` tokens sorted
- `name_tok1`: first `name_n2` token
- `addr_n1`: normalised address
- `addr_n2`: sorted set of expanded address tokens
- `addr_digits`: sorted set of digit groups
- `postal`, `house`, `country`

**Channel 1: 16 exact-key equality joins**, run per source (S2 and S3 separately). A join block with more than 100
pool records is dropped entirely (cap 100). The keys are:
- `name_n1`, `name_n2`, `name_n3`
- `addr_n1`, `addr_n2`
- `name_n2+country`, `addr_n2+country`
- `name_n1+addr_n1`, `name_n2+addr_n2`, `name_n3+addr_n2`
- `name_n3+house`, `name_tok1+house`, `name_tok1+addr_digits`
- `postal+name_tok1`, `postal+house`
- `house+addr_digits`

Exact joins search the whole pool of the source. They are restricted to the same country only when the key
contains `country`.

**Channel 2: TF-IDF cosine top-K retrieval** (`ber/blocking.py`, numba inverted index):
- Binary term frequency; smooth IDF from the pool, `log((N+1)/(df+1))+1`.
- Two channels, each giving **top 10 per source**, so at most 20 per channel per S1:
  - `nameaddr`: name words, character trigrams of the joined name, address words, digit groups and house number;
  - `addr`: address tokens only.
- **Postings budget:** each query scans its rarest tokens first, until 20,000 postings have been read. The rarest
  token is always used, so the cost per query is bounded.
- **Country partitioning:** retrieval runs within the query's country, because training has 0 cross-country true
  pairs. **Fallback:** a query whose country is empty or absent from that source's pool is searched against the
  whole pool of that source.

**Combination:**
- The final candidate set is the deduplicated union (`np.union1d`) of all 18 channel outputs (16 exact + 2
  retrieval), for S2 and S3.
- Candidates are not ranked or filtered after the union. The union is exactly the set scored by the matcher, and
  exactly what `candidate_pairs.tsv` contains.
- There are no hard-coded candidates and no external lookups.

**Measured results:**
- On the held-out validation split: candidate recall 98.214%, 25,826,498 pairs (58.5 per S1).
- On test: 126,274,337 pairs for 1,732,544 S1; mean 72.9, median 42 and maximum 533 per S1; no S1 without
  candidates.

**Evaluated but NOT used in production:**
- a `name`-only retrieval channel
- retrieval with K = 50
- a second `nameaddr` channel with a 200,000-postings budget (analysed after submission, never applied)

## 7. Feature Engineering
`ber/match_features.py` builds **50 float32 features** per (S1, candidate) pair. The same code is used for training
and test.
- `is_s3`: the candidate's source.
- `eq_<key>` for the 11 keys above (0 if either side is empty), plus the presence indicators `postal_both`,
  `house_both` and `digits_both`.
- Target flags (7): name missing, address missing, non-Latin name, domain-like name, junk prefix, postal present,
  landmark in address.
- Query flags (2): postal present, landmark in address.
- Name token-set similarities (8) over words and character trigrams: query size, target size, intersection,
  Jaccard, containment in both directions, IDF cosine and IDF-weighted Jaccard.
- The same 8 similarities over address tokens.
- `nameaddr_cos`: IDF cosine over all tokens.
- **Query-side candidate context** (label-free): the rank of the pair's `nameaddr_cos`, its gap to the best score,
  and the candidate count. These are computed within the S1's candidates from the same source (`rank_src`,
  `gap_src`, `n_cand_src`) and from both sources (`rank_all`, `gap_all`, `n_cand_all`).
- **Target-side candidate context:** the same rank, gap and count among all S1 that retrieved this target (`t_rank`,
  `t_gap`, `t_ncand`).

IDF is always computed from the pool being matched: the full training pool for training, and the full test pool for
test.

## 8. Model
`ber/gbdt.py` is a histogram GBDT for binary log-loss, written from scratch in numba.

**Hyperparameters** (`config/matcher_config.json`):
- 300 trees, depth-wise growth to max depth 6
- learning rate 0.1, L2 λ = 1.0, minimum child hessian 20
- at most 64 quantile bins per feature, with edges from a seeded 400,000-row sample (seed 0)
- base score = log-odds of the weighted positive rate
- no row or feature subsampling

**Training rows:**
- All training S1 are blocked against the full training pool: 160,550,236 candidate pairs, of which 7,305,755 are
  positive.
- Every positive pair is kept. Negatives are sampled deterministically with the hash rule
  `((code·0x9E3779B97F4A7C15) >> 40) % 1000 < 100`, which keeps 10%, and get weight 10.
- This gives 22,629,394 rows.

**Decision rule:**
- A pair is a match if its probability is **≥ 0.69**.
- **One-S1-per-target:** when several S1 claim the same S2/S3 record, only the highest-probability S1 keeps it. In
  training, no S2/S3 record belongs to more than one S1.
- Model type, tree count (300), threshold (0.69) and the rule were selected by official macro F0.5 on an internal
  split of the training fold. They were then confirmed on a held-out validation split (macro F0.5 0.962847), and
  were never tuned on test.

## 9. Output Files
Both files are UTF-8 TSV with LF line endings and one row per test S1, in `test_source1.tsv` order. IDs within a
row are sorted and comma-separated, and an empty list is written as `S1-x<TAB>`.
- **`matching_results.tsv`:** header `source1_entity_id<TAB>matched_entity_ids`. It holds the final matches:
  5,503,470 pairs; 110,945 S1 have no match.
- **`candidate_pairs.tsv`:** header `source1_entity_id<TAB>candidate_entity_ids`. It holds the exact candidate set
  the GBDT scored: 126,274,337 pairs. Every predicted match is a member of it.

## 10. Reproducibility Notes
**Provenance:**
- The submitted outputs were produced by the project's experiment driver, `experiments/final_pipeline.py`.
  `src/run_pipeline.py` consolidates that driver.
- It calls the **same** `ber` modules (byte-identical) with the same configuration. The candidate union is the same
  as the driver's, without the recall bookkeeping.
- On a small labelled subset, `run_pipeline.py` and the original driver produced **byte-identical**
  `matching_results.tsv` and `candidate_pairs.tsv`. This was re-checked in a clean venv built from
  `requirements.txt`.
- The full-scale run was **not** repeated with `run_pipeline.py`, because it takes about 3 hours. The submitted
  files are the originals.

**Determinism:**
- No step is random apart from seeded sampling.
- Negative sampling is a fixed hash; GBDT binning uses seed 0, with no subsampling.
- Repeated runs on the same machine gave identical outputs.
- The numba kernels accumulate floating-point histograms in parallel. Bit-identical probabilities on a different
  CPU or thread count are therefore expected but not guaranteed. A pair lying exactly at the 0.69 threshold could
  in principle flip.

**Resources** (measured on the full data, 16-thread laptop CPU, Windows 11):
- **Runtime:** about 3.2 hours end to end.
  - featurisation: about 15 minutes
  - training blocking: about 44 minutes
  - test blocking: about 55 minutes
  - cosine and target-context passes: about 34 minutes
  - training rows: about 9 minutes
  - GBDT training: about 18 minutes
  - test inference: about 13 minutes
  - writing outputs: about 2 minutes
- **Memory:** everything is chunked and memory-mapped. Peak working set was about 8 GB, and private memory about
  2–4 GB. Allow at least 8 GB of free RAM.
- **Disk:** about 20–25 GB of intermediate files in `--work-dir`. The `--work-dir` help text says about 40 GB, as a
  conservative upper bound.

**Order and paths:**
- Training (phases 2–7) must finish before inference (phases 12–13). `run_pipeline.py --phase all` enforces this
  order.
- Every path comes from the command line or is relative to `src/`. There are no machine-specific paths.

## 11. Compliance
- **Data:** only the provided challenge files are read (the `--train-dir`/`--test-dir` TSVs). Test labels do not
  exist and are not used.
- **No external resources:** the code makes no network access, API calls, geocoding, business registries,
  external databases or downloads, and does no data augmentation.
  - The source contains no `requests`, `urllib`, `socket` or `http` client usage.
  - The only `http`/`www` strings are regex patterns that flag domain-like business *names*.
- **Hard-coded lists:** the only ones are generic text-normalisation lists: legal-form words, street/state
  abbreviations and landmark words.
- **Models:** no pretrained models, embeddings or weights. The only model is the GBDT trained here from the
  training labels: 300 trees of depth ≤ 6 (≤ 127 nodes each). It is a few hundred KB, far below the 8B-parameter
  limit.
- **Libraries:** all dependencies are under permissive open-source licences: numpy (BSD-3-Clause), scipy
  (BSD-3-Clause), numba (BSD-2-Clause) and llvmlite (BSD-2-Clause; bundles LLVM under Apache-2.0 with LLVM
  exceptions).
