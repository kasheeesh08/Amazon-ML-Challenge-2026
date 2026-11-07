# Blocking / candidate-recall gap investigation (diagnostic only)

Scripts: `experiments/diagnostics/blocking_gap/blocking_gap.py` (writes `results.json` and `run.log`) and `hybrid.py` (writes `hybrid.json` and `hybrid.log`).

**Scope:** saved artifacts only. Nothing was re-blocked, retrained or re-inferred, and nothing was uploaded. Test labels do not exist, so **no test candidate recall is claimed**.

## Artifacts inspected

- **Configuration:** `experiments/blocking/blocking_config.json`, the hash recorded in `experiments/final/test_inputs.json`, and `submission_package/.../src/config/blocking_config.json`.
- **V, normal validation** (validation S1 → validation pool): `experiments/matching/cand_val.npy`, and `X_val.npy` for the name+address cosine.
- **T, test-ratio** (validation pool plus random fit-fold records; S1 per pool record = test's 0.174; pool 2.54M): `experiments/diagnostics/trank/cand_T.npy`, `X_T.npy`, `poolT_s{2,3}.npy`, `candidates_T.json`.
- **S, stress** (full 10.32M training pool): `experiments/test_scale/stress/cand_stress.npy`, `candidates.json`, and `experiments/diagnostics/trank/X_S.npy`.
- **TEST, real and unlabelled** (1,732,544 S1; pool 9.97M): `experiments/test_scale/cand_test.npy`, `cos_test.npy`, `candidates.json` (channel pair counts), and `features_test/` (country, names, flags).
- **Other:** `experiments/blocking/checkpoints/validation_channels.json`, `experiments/blocking/BLOCKING_REPORT.md`, `experiments/diagnostics/trank/evaluate.json` and `p_T_A.npy`, `experiments/TEST_SCALE_REPORT.md`, and the LOCO, France-threshold and target-context-ablation reports.

## 1. Baseline blocking configuration (unchanged)

The SHA-256 is `91a3897685414bb48c4bc4f00a162e1b245eb8640166f950a76bdc379042e0f5`, identical in all three places listed above.

- **TF-IDF retrieval, run per source (S2 and S3 separately) and only within the query's country:**
  - name+address tokens, top 10 per source, up to 20,000 postings
  - address tokens, top 10 per source, up to 20,000 postings
- **16 exact-key joins:** name_n1, name_n2, name_n3, addr_n1, addr_n2, name_n2+country, addr_n2+country, name_n1+addr_n1, name_n2+addr_n2, name_n3+addr_n2, name_n3+house, name_tok1+house, name_tok1+addr_digits, postal+name_tok1, postal+house, house+addr_digits.
  - Each uses a block cap of 100: any key shared by more than 100 pool records is dropped.
- The candidate set is the union of all channels.

## 2. Candidate counts per S1

| Env | Pool records per S1 | Total pool | Mean | Median | p90 | p95 | p99 | Max | Zero | Exactly 1 | ≥ 200 | ≥ 300 | ≥ 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V | 4.68 | 2.06M | 58.5 | 39 | 123 | 160 | 208 | 390 | 0 | 0 | 6,831 | 315 | 0 |
| T | 5.75 | 2.54M | 59.7 | 39 | 125 | 164 | 213 | 390 | 0 | 0 | 7,558 | 478 | 0 |
| S | 23.4 | 10.32M | 72.8 | 42 | 160 | 192 | 250 | 437 | 0 | 0 | 17,758 | 1,269 | 10 |
| **TEST** | **5.75** | **9.97M** | **72.9** | **42** | **153** | **188** | **255** | **533** | 0 | 0 | 66,411 | 5,194 | 130 |

By country:

| Env / country | Mean | Median | p99 | Max | Pool per S1 |
|---|---:|---:|---:|---:|---:|
| V US / India | 64.2 / 50.0 | 39 / 39 | 214 / 197 | 390 / 363 | 4.67 / 4.68 |
| T US / India | 65.5 / 51.1 | 39 / 40 | 222 / 187 | 390 / 365 | 5.75 / 5.76 |
| S US / India | 71.0 / 75.5 | 40 / 46 | 249 / 252 | 433 / 437 | 23.4 / 23.4 |
| TEST US / India / France | 67.1 / 77.0 / 74.6 | 40 / 47 / 43 | 225 / 255 / 292 | 425 / 487 / 533 | 5.76 / 5.82 / 5.53 |

- **Minimum is 20 candidates in every environment.** The two retrieval channels always return 10 per source. No S1 has zero or exactly one candidate.
- **The real test's candidate distribution matches S, not T.**
  - Exact-name channels produce about as many pairs per S1 on test as in S. name_n2: 18.5 on test vs 18.6 in S vs 10.9 in T. name_tok1+house: 10.9 on test vs 8.0 in S vs 3.5 in T.
  - Blocking depends on **absolute** pool size (block cap, top-K competition), and the test pool (9.97M) is S-sized.
  - Target context, by contrast, depends on the S1-to-pool **ratio**, which is T-like.

## 3. Candidate recall (labelled environments only)

| Metric | V | T | S |
|---|---:|---:|---:|
| **Candidate recall** | 98.214% | 97.991% | **95.631%** |
| S2 / S3 | 98.15% / 98.28% | 97.92% / 98.06% | 95.53% / 95.72% |
| US / India | 98.85% / 97.27% | 98.67% / 96.98% | 96.47% / 94.38% |
| Matched S1 with all / at least one true match retrieved | 94.51% / 99.83% | 93.85% / 99.80% | 87.22% / 99.54% |
| True pairs missed | 27,277 | 30,686 | 66,750 |
| **Oracle macro F0.5** (a perfect matcher on these candidates) | 0.99386 | 0.99305 | 0.98453 |
| Oracle US / India | 0.99647 / 0.98994 | 0.99590 / 0.98878 | 0.98854 / 0.97852 |
| Current model: missed pairs never in the candidates / all its FN | 27,277 / 90,476 | 30,686 / 89,791 | 66,750 / 104,785 |

- **By match count:** recall is flat. S1 with 1 true match: 97.9% (V), 95.3% (S); S1 with 2–3, 4–5 and 6+: 98.2% (V), 95.6% (S).
- **Singletons** have no true pairs, so recall doesn't apply to them. For them, blocking only supplies candidates the matcher must reject (mean 60.6 in V, 74.8 in S; BLOCKING_REPORT).
- **France:** no labels, so its recall is UNRESOLVED.

**Recall by true-pair category:**

| Category | Share of true pairs | Recall V | Recall T | Recall S | Share of S misses |
|---|---:|---:|---:|---:|---:|
| House number differs (both present) | 17.6% | 95.26% | 94.75% | **88.93%** | 44.7% |
| Target name non-Latin | 7.2% | 89.26% | 88.35% | **80.75%** | 31.9% |
| Target address empty | 4.4% | 90.02% | 88.75% | **78.51%** | 21.6% |
| Target name domain-like | 5.2% | 98.08% | 97.66% | 91.72% | 9.8% |
| Target name junk prefix | 2.4% | 98.96% | 98.73% | 95.91% | 2.3% |
| Sorted name differs (`name_n3`) | 47.4% | 96.23% | 95.76% | 91.03% | 97.3% |

**Recall for generic vs unique S1 names:**
- "Generic" means the S1's normalised name is shared by at least 10 S1 of the same country *in that environment*.
- In S: US 92.93% generic vs 97.23% unique; India **88.06%** generic vs 97.34% unique.
- In V: US 96.88% vs 99.26%; India 93.54% vs 98.99%.

## 4. Channels: recall of each channel on its own, and true pairs only that channel finds

| Channel | V recall | T recall | S recall | S unique true pairs | TEST pairs per S1 | S pairs per S1 | T pairs per S1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieval nameaddr@10 | 93.97% | 93.82% | **90.74%** | 53,751 | 20.0 | 20.0 | 20.0 |
| retrieval addr@10 | 87.24% | 86.49% | **79.88%** | 24,260 | 20.0 | 20.0 | 20.0 |
| exact house+addr_digits (cap 100) | 36.44% | 35.02% | **25.77%** | 1,241 | 13.7 | 15.5 | 19.6 |
| exact name_tok1+house | 51.84% | 51.84% | 51.73% | 1,540 | 10.9 | 8.0 | 3.5 |
| exact name_tok1+addr_digits | 48.40% | 48.40% | 48.37% | 1,871 | 7.4 | 4.7 | 2.4 |
| exact name_n3 | 52.50% | 52.23% | 49.66% | 1,832 | 18.8 | 18.6 | 11.2 |
| exact name_n1 | 25.69% | 25.69% | 25.32% | 1,319 | 6.8 | 8.8 | 3.4 |

- The other exact channels contribute at most 14 unique true pairs each (see `results.json`).
- The losses at full pool size come from **retrieval competition** (top-10 against 5× more records) and **dropped blocks** (house+addr_digits blocks exceeding the cap of 100).

## 5. Real test: signs of blocking problems (label-free)

| Check | US | India | France | Reading |
|---|---:|---:|---:|---|
| Candidate explosion: S1 with ≥ 300 candidates | 766 (0.12%) | 2,253 (0.28%) | 2,175 (**0.84%**) | France has the heaviest tail (p99 292, max 533). S has 0.29%. |
| Unusually low counts | minimum 20 | minimum 20 | minimum 20 | None; retrieval guarantees 20. |
| Generic-name collisions: S1 whose name is shared by ≥ 10 / ≥ 100 S1 | 16.8% / 5.5% | 32.8% / 0.2% | 19.9% / 4.2% | Generic names get about 2.5–3× the candidates of unique names (France 154 vs 50). |
| Address-based failure: pool records with an empty address (test vs train) | 2.9% vs 3.6% | 2.4% vs 3.0% | 3.0% (no train pool) | No harder than train. |
| Pool with non-Latin names | 0.0% | 18.4% (train 18.2%) | 0.0% | India the same as train; France has none. |
| Pool with a postal code | 2.9% | 2.0% | **0.27%** | The postal channels (0.6% recall) are useless for France; minor. |
| TF-IDF retrieval: best candidate cosine, p1 / p5 | 0.656 / 0.765 | 0.575 / 0.680 | 0.687 / **0.795** | France S1 find strongly similar candidates; no sign of retrieval failure. |
| S1 whose best candidate cosine < 0.3 | 0.01% | 0.01% | 0.00% | Near zero everywhere. In V such S1 are 98.7% singletons, so this can't detect partial misses. |

## 6. Estimated blocking cost at test-sized pool (`hybrid.json`)

The estimate takes the current model's predictions in T, where the target context resembles test, and removes every true pair that the unchanged blocking fails to retrieve against the full S-sized pool.

| | Macro F0.5 | Pair precision | Pair recall | US | India |
|---|---:|---:|---:|---:|---:|
| T, current model (measured) | 0.95432 | 97.22% | 94.12% | 0.96426 | 0.93942 |
| **T matcher + S blocking losses** | **0.94804** | 97.17% | 92.55% | 0.95828 | 0.93270 |
| **Blocking cost at test-sized pool** | **−0.0063** | | −1.57 pp | −0.0060 | −0.0067 |

23,947 of T's correct predictions are pairs that S-sized blocking never produces.

## 7. Comparison with the previous diagnostics: updated gap accounting

| Step | Estimate | Source |
|---|---:|---|
| Local validation | 0.9628 | Step 3 |
| + test S1-to-pool ratio (orphans / target context) | 0.9543 (−0.0085) | t_rank diagnostic (T) |
| + test-sized pool (blocking) | 0.9480 (−0.0063) | this report |
| + test country mix (US 45% / India 55% of non-France) | 0.9442 (−0.0038) | this report's per-country hybrid values |
| **France needed to reach 0.924** | **about 0.81** | arithmetic |
| France at a LOCO-sized penalty (about −0.075) | overall about 0.933, leaving about **0.009 unexplained** | LOCO |

## 8. Evidence FOR blocking being part of the gap

- Recall degrades measurably at test-sized pools: 98.21% → 95.63%. The oracle ceiling drops 0.0085, and the hybrid estimate costs **0.0063 F0.5**.
- The real test's candidate distribution and exact-channel volumes match the full-pool environment S, where this loss was measured.
- The losses concentrate in generic names (India generic 88.1%), non-Latin targets (80.8%), empty-address targets (78.5%) and house-number differences (88.9%), with retrieval competition and cap-dropped exact blocks as the mechanism.
- France has the heaviest candidate tail and generic-name structure similar to US (4.2% of S1 share their name with 100+ others). Its recall is unmeasurable.

## 9. Evidence AGAINST blocking being the main cause

- Even at S-sized pools, blocking costs only about **0.006** of the roughly 0.039 gap (about 16%). A perfect matcher on S's candidates would still score 0.985.
- On the real test there is no candidate starvation (minimum 20), no S1 without a plausible candidate, and pool composition no harder than train (empty addresses, non-Latin shares).
- France's retrieval similarity is the highest of the three countries.
- The matcher, not blocking, accounts for most of the error in every labelled environment. Missed candidates are 30% of the current model's false negatives in V and 64% in S, but false positives dominate the F0.5 loss in S.

## 10. Limitations from the lack of test labels

- **No test candidate recall** can be computed. The 0.0063 figure assumes test blocking behaves like validation S1 against the full *training* pool.
- France's recall, and its categories (e.g. French name/address variants not seen in training), are unknown.
- The generic-name shares in V/T/S are computed within 441k validation S1, and on test within up to 810k S1 per country. Frequency counts grow with population size, so those shares are not strictly comparable.
- The best-candidate cosine only detects S1 with *no* good candidate, not partial misses.

## 11. Should blocking be modified, and the next experiment

- **Blocking is healthy but not free at test scale.** It is a real, quantified contributor (about −0.006), not the main source of the gap.
- **Do not modify production blocking now.** The maximum recoverable amount is bounded by the S−T difference in recall (about 2.4 pp of true pairs, 0.006–0.009 F0.5), and any change also adds candidates the matcher must reject.
- **Smallest justified controlled experiment**, if the team wants to pursue it: measure *candidate recall only* in environment S (validation S1 vs full training pool) for **nameaddr retrieval K = 20 per source**, instead of 10, changing nothing else. That is the channel with the most unique true pairs (53,751) and the known @10 → @50 headroom (93.97% → 96.30% on V).
  - Report the union recall gain, overall and for generic names, non-Latin targets and empty addresses, plus the extra candidates per S1.
  - Only if the union recall gain is at least about 1 pp should end-to-end F0.5 be evaluated. That would need matcher features for the new candidates, which is a larger step.
- **The main unexplained component remains France** (it would need about 0.81 to reach 0.924) plus about 0.009 of residual that no diagnostic has located. Blocking cannot explain France's share without France labels.

## Integrity

- `output/matching_results.tsv`: `84ede6d4f2e44ab4469ea7798a95fbd8b3c49fc0f3f1561d47064c1caaa0e807`, unchanged.
- `output/candidate_pairs.tsv`: `1842a3df05f69fd9236e4601cef27ded0d18e7e2c481dd803f27e9f8ff4835ea`, unchanged.
- `experiments/final/`: all 168 files hash-identical before and after. The blocking config is unchanged (`91a38976…0e5f5`). All 34 protected files are unchanged (`results.json` → integrity).
- Files created, all in `experiments/diagnostics/blocking_gap/`:
  - `blocking_gap.py`
  - `hybrid.py`
  - `hashes_before.json`
  - `results.json`
  - `hybrid.json`
  - `run.log`
  - `hybrid.log`
  - `BLOCKING_GAP_REPORT.md`
- Cost: `blocking_gap.py` took 226 s, including hashing; `hybrid.py` took about 1 minute. No training or inference.
