"""Business Entity Resolution (Amazon ML Challenge 2026) — project package.

Modules
-------
ids_tsv           read/write the competition's ID-list TSV files (mirrors utils/validate_submission.py)
metrics           the official per-entity F0.5 and its macro average
validation_split  leakage-safe 80/20 split of the training data at the Source-1 entity level
"""
