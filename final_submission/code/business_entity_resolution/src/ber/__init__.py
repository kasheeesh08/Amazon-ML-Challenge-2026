"""Business Entity Resolution (Amazon ML Challenge 2026) — packaged solution.

Modules
-------
normalize       text normalisation, exact keys, blocking tokens, record flags
features        streaming, chunked, multi-process featurisation of a source TSV into an on-disk store
blocking        candidate generation: exact-key channels + TF-IDF top-K retrieval (numba)
match_features  the 50 pairwise features of the matcher
gbdt            histogram gradient-boosted trees (numba)
matcher         decision rule (threshold + one-S1-per-target), TSV writer (and the unused LogReg baseline)
ids_tsv         competition ID-list TSV format helpers
pipeline        end-to-end orchestration helpers (chunked candidate union, target context, ground truth)
"""
