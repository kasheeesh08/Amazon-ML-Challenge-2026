"""Read and write the competition's ID-list TSV files.

Covers ``matching_results.tsv``, ``candidate_pairs.tsv`` and ``train_ground_truth.tsv``: a header row,
then one ``<source1_entity_id>\\t<comma-separated IDs>`` row per Source-1 entity.

Parsing deliberately mirrors ``utils/validate_submission.py`` (the official validator) line for line:

* file opened in text mode, UTF-8, universal newlines (so CRLF is tolerated exactly as the validator
  tolerates it);
* header compared after ``strip().lower()`` of each column;
* a row is split with ``line.partition("\\t")``; a row without a TAB is malformed unless blank;
* the ID list is ``rest.rstrip("\\n").split(",")`` when ``rest.strip()`` is non-empty, else empty.
  IDs are **not** stripped, so ``"S2-1, S2-2"`` yields ``" S2-2"`` which fails the S2-/S3- prefix rule
  — exactly as in the validator.

Rules checked (all are rejections per the problem statement p.3/p.5 and the validator): duplicate S1
rows, duplicate IDs within a list, Source-1 IDs in a list (self-matches), IDs without an S2-/S3-
prefix, missing required S1 rows, S1 rows not in the required set and — when ``valid_ids`` is given —
IDs that do not exist in the Source-2/3 files.
"""

import os

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
GROUND_TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]  # same columns as matching


class FormatError(ValueError):
    """Raised in strict mode; ``issues`` holds every problem found."""

    def __init__(self, path, issues):
        self.path = path
        self.issues = list(issues)
        super().__init__(f"{path}: {len(self.issues)} format issue(s): " + " | ".join(self.issues[:5]))


def read_source_ids(path):
    """Set of first-column entity IDs of a source TSV (same logic as the validator's ``read_ids``)."""
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def _examples(items, k=5):
    items = sorted(items)
    shown = ", ".join(items[:k])
    return f"{len(items)} total, e.g. {shown}, ..." if len(items) > k else shown


def read_id_list_tsv(path, expected_header, required=None, valid_ids=None, strict=True):
    """Parse an ID-list TSV.

    Returns ``(mapping, issues)`` where ``mapping`` is ``{source1_id: frozenset(ids)}``.
    ``required``: set of S1 IDs that must each appear exactly once (None = no check).
    ``valid_ids``: set of existing S2/S3 IDs (None = skip the existence check).
    In strict mode any issue raises :class:`FormatError`.
    """
    issues = []
    if not os.path.isfile(path):
        issues.append(f"file not found: {path}")
        if strict:
            raise FormatError(path, issues)
        return {}, issues

    mapping = {}
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            issues.append("file is empty")
            if strict:
                raise FormatError(path, issues)
            return {}, issues
        if DELIM not in header and "," in header:
            issues.append("header has no TAB but contains commas — file looks comma-separated")
            if strict:
                raise FormatError(path, issues)
            return {}, issues
        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != list(expected_header):
            issues.append(f"unexpected header {cols}; expected {list(expected_header)}")
            if strict:
                raise FormatError(path, issues)
            return {}, issues

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    issues.append(f"malformed row (no tab) at line {line_num}: {line.rstrip()!r}")
                continue
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = frozenset(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    checks = [
        (dup_rows, "duplicate source1_entity_id row(s): {}"),
        (intra_dupes, "repeated ID inside a list for: {}"),
        (self_matches, "Source-1 IDs (self-matches) in a list: {}"),
        (wrong_prefix, "IDs without an S2-/S3- prefix: {}"),
        (unknown, "IDs not present in the Source-2/3 files: {}"),
    ]
    if required is not None:
        checks += [
            (set(required) - seen, "required S1 entity(ies) missing: {}"),
            (seen - set(required), "S1 ID(s) not in the required set: {}"),
        ]
    for offenders, msg in checks:
        if offenders:
            issues.append(msg.format(_examples(offenders)))
    if strict and issues:
        raise FormatError(path, issues)
    return mapping, issues


def write_id_list_tsv(path, mapping, header, s1_order=None):
    """Write ``{source1_id: iterable_of_ids}`` as a TAB-separated, UTF-8, LF file.

    Empty lists are written as ``S1-x\\t`` (the validator requires the TAB). IDs are sorted for
    deterministic output. Refuses IDs containing separators and duplicate IDs within a list.
    """
    order = list(mapping) if s1_order is None else list(s1_order)
    if len(order) != len(set(order)):
        raise ValueError("duplicate source1 IDs in s1_order")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(DELIM.join(header) + "\n")
        for s1 in order:
            ids = list(mapping.get(s1, ()))
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate IDs in list for {s1}")
            for x in ids:
                if not x or any(c in x for c in ",\t\n\r "):
                    raise ValueError(f"invalid ID {x!r} for {s1}")
            f.write(f"{s1}{DELIM}{','.join(sorted(ids))}\n")
