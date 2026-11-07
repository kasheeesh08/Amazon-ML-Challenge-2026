#!/usr/bin/env python3
"""
Exploratory data analysis (EDA) for the Business Entity Resolution challenge.

READ-ONLY with respect to dataset/, utils/, README.md, Documentation_template.md and docs/.
Writes only inside experiments/:

    experiments/_eda_cache/       per-record 64-bit hashes of normalised keys (.npy) and
                                  per-file profiling stats (.json); reused on re-runs
    experiments/eda_results.json  every measured number
    experiments/EDA_REPORT.md     human-readable report generated from those numbers

Design
------
* Every TSV is streamed line by line in binary mode (split on b"\\n" only) — never loaded whole.
* Each record is reduced to a few 64-bit hashes of normalised keys (name, address, postal code,
  house number, country, ...). numpy then groups/joins those integer arrays.
* No Cartesian product is ever materialised. The number of candidate pairs a blocking key would
  produce is computed as  sum_k  count_S1(k) * count_S2S3(k).
* Recall of a key = fraction of ground-truth (S1, S2/S3) pairs whose keys are equal and non-empty.
  Because the training ground truth lists *all* matches of every S1 entity, every other pair is a
  true non-match, so  precision = recovered true pairs / candidate pairs  is exact.

Analysis only: no model is trained, nothing is downloaded, no external data or API is used.
Normalisation rules here are deliberately simple EDA heuristics, not the final pipeline.

Usage (from the project root):
    python experiments/eda.py [--workers 4] [--rebuild] [--sample-pairs 30000]
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from array import array
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Overridable via environment (inherited by spawned workers) — used only for smoke tests.
DATA = Path(os.environ.get("EDA_DATA_DIR", ROOT / "dataset"))
OUT = Path(os.environ.get("EDA_OUT_DIR", ROOT / "experiments"))
CACHE = OUT / "_eda_cache"
NORM_VERSION = 2
SEED = 42

SOURCE_FILES = {
    "train_s1": "train/train_source1.tsv",
    "train_s2": "train/train_source2.tsv",
    "train_s3": "train/train_source3.tsv",
    "test_s1": "test/test_source1.tsv",
    "test_s2": "test/test_source2.tsv",
    "test_s3": "test/test_source3.tsv",
}
GT_FILE = "train/train_ground_truth.tsv"

# Hashed per-record keys stored in the cache (plus the numeric "id").
HASH_COLS = [
    "name_raw", "name_n1", "name_n2", "name_n3", "name_tok1",
    "addr_n1", "addr_n2", "addr_digits", "postal", "house", "country",
]

# Blocking keys evaluated against the ground truth (tuples = conjunction of fields).
KEY_SPECS = [
    ("name_raw",), ("name_n1",), ("name_n2",), ("name_n3",), ("name_tok1",),
    ("addr_n1",), ("addr_n2",), ("addr_digits",), ("postal",), ("house",), ("country",),
    ("name_n1", "country"), ("name_n2", "country"), ("name_n3", "country"),
    ("addr_n1", "country"), ("addr_n2", "country"),
    ("name_n1", "addr_n1"), ("name_n2", "addr_n2"), ("name_n3", "addr_n2"),
    ("name_n1", "addr_n1", "country"),
    ("name_n2", "postal"), ("name_n2", "house"), ("name_n3", "house"),
    ("name_tok1", "postal"), ("name_tok1", "house"), ("name_tok1", "country"),
    ("name_tok1", "addr_digits"), ("postal", "house"), ("postal", "country"),
    ("house", "country"), ("addr_digits", "country"),
]

KEY_DESCRIPTIONS = {
    "name_raw": "business_name, whitespace-stripped only (case/punctuation kept)",
    "name_n1": "name casefolded, accents stripped, punctuation->space, '&'->'and'",
    "name_n2": "name_n1 + single-letter runs joined (l l c->llc) + legal-suffix tokens removed",
    "name_n3": "name_n2 tokens sorted (word-order insensitive)",
    "name_tok1": "first token of name_n2",
    "addr_n1": "address casefolded, punctuation->space (order kept)",
    "addr_n2": "address tokens with street/state abbreviations expanded, as a sorted token set",
    "addr_digits": "sorted set of all digit groups in the address",
    "postal": "5-digit ZIP/France code or 6-digit (or 3+3) Indian PIN, heuristic extraction",
    "house": "first house/building number in the address (not the postal code)",
    "country": "country label casefolded",
}

PLACEHOLDERS = {"nan", "none", "null", "n/a", "na", "-", "--", "unknown", "0", ".", "?"}

LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "plc", "pllc", "pc", "lc", "the",
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "gmbh",
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "कंपनी",
}

ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "ct": "court",
    "pl": "place", "sq": "square", "pkwy": "parkway", "cir": "circle", "ter": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite", "r": "rue",
    "mg": "mahatma gandhi", "ngr": "nagar", "clny": "colony", "dist": "district", "no": "number",
    "bldg": "building", "mkt": "market", "extn": "extension", "ext": "extension",
}
ABBR = {k: v for k, v in ABBR.items() if " " not in v}

STATE_ABBR = {
    # US
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia",
    # India (only codes that do not clash with the US list above)
    "dl": "delhi", "hr": "haryana", "mh": "maharashtra", "kl": "kerala", "up": "uttar pradesh",
    "wb": "west bengal", "rj": "rajasthan", "gj": "gujarat", "pb": "punjab", "br": "bihar",
    "ts": "telangana", "tg": "telangana", "ap": "andhra pradesh", "mp": "madhya pradesh",
    "ka": "karnataka", "jh": "jharkhand", "uk": "uttarakhand", "hp": "himachal pradesh",
    "cg": "chhattisgarh", "jk": "jammu and kashmir", "py": "puducherry", "as": "assam",
}


# --------------------------------------------------------------------------------------
# Normalisation helpers (shared by the streaming workers and the in-memory analysis)
# --------------------------------------------------------------------------------------

def _build_punct_table():
    table = {}
    for cp in range(0x30000):
        if unicodedata.category(chr(cp))[0] not in "LNM":
            table[cp] = " "
    for ch in "'’‘`´":
        table[ord(ch)] = None
    table[ord("&")] = " and "
    return table


PUNCT_TABLE = _build_punct_table()
COMBINING_RE = re.compile("[̀-ͯ]")  # Latin/Greek/Cyrillic diacritics only
POSTAL_RE = re.compile(r"(?<![0-9A-Za-z])([0-9]{3} ?[0-9]{3}|[0-9]{5})(?:-[0-9]{4})?(?![0-9A-Za-z])")
WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]{2,}")
ALNUM_RE = re.compile(r"[0-9A-Za-z]")
LEADING_ZERO_RE = re.compile(r"(?<![0-9])0+(?=[0-9])")


def strip_leading_zeros(v):
    return LEADING_ZERO_RE.sub("", v)
HOUSE_RE = re.compile(r"(?<![0-9A-Za-z])([0-9]+[A-Za-z]?(?:\s?[/-]\s?[0-9]+[A-Za-z]?)*)")
DIGITS_RE = re.compile(r"\d+")
NONDIGIT_RE = re.compile(r"\D")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")
DOMAIN_RE = re.compile(
    r"(?i)(\bwww\.|https?://|\.(com|net|org|in|co|io|biz|info|fr|us|shop|store|online)\b)")
DBA_RE = re.compile(r"(?i)(\bd/b/a\b|\bdba\b|\ba/k/a\b|\baka\b|\bt/a\b|trading as|doing business as)")
LANDMARK_RE = re.compile(
    r"(?i)\b(near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|landmark)\b")
UNIT_RE = re.compile(r"(?i)(\b(suite|ste|apt|apartment|unit|floor|fl|flat|room|rm)\b|#\s?\d)")
POBOX_RE = re.compile(r"(?i)\bp\.?\s?o\.?\s?box\b")
STATEFIRST_RE = re.compile(r"^\s*[A-Za-z]{2}\s*,")
COUNTRYNAME_RE = re.compile(r"(?i)\b(usa|united states|india|france)\b")

_SCRIPT_CACHE = {}


def basic(s):
    """Casefold, strip Latin diacritics, punctuation -> space, collapse whitespace."""
    if not s:
        return ""
    if not s.isascii():
        s = unicodedata.normalize("NFKD", s)
        s = COMBINING_RE.sub("", s)
        s = unicodedata.normalize("NFKC", s)
    s = s.casefold().translate(PUNCT_TABLE)
    return " ".join(s.split())


def collapse_initials(tokens):
    """Join runs of single letters: ['l','l','c'] -> ['llc'], ['a','b','c','plumbing'] -> ['abc', ...]."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


def extract_postal(addr):
    """Return (normalised_postal, raw_match). Last plausible postal code wins."""
    best, best_raw = "", ""
    for comp in addr.split(","):
        for m in POSTAL_RE.finditer(comp):
            tok = m.group(1).replace(" ", "")
            if len(tok) == 5 and not ALNUM_RE.search(comp[:m.start()]):
                # "17560 Ellis Road": 5-digit number opening a street component = house number
                if len(WORD_RE.findall(comp[m.end():])) >= 2:
                    continue
            best, best_raw = tok, m.group(0)
    return best, best_raw


def extract_house(addr, postal):
    for m in HOUSE_RE.finditer(addr):
        v = re.sub(r"\s", "", m.group(1)).lower()
        if postal and NONDIGIT_RE.sub("", v) == postal:
            continue
        return strip_leading_zeros(v)  # "0617" -> "617", "3/05" -> "3/5"
    return ""


def char_script(c):
    s = _SCRIPT_CACHE.get(c)
    if s is None:
        nm = unicodedata.name(c, "")
        s = nm.split(" ", 1)[0] if nm else "UNKNOWN"
        _SCRIPT_CACHE[c] = s
    return s


def scripts_of(s):
    if s.isascii():
        return {"LATIN"} if LATIN_ALPHA_RE.search(s) else set()
    return {char_script(ch) for ch in s if ch.isalpha()}


def normalize_record(name, addr, country):
    name = name or ""
    addr = addr or ""
    n1 = basic(name)
    toks = collapse_initials(n1.split())
    t2 = [t for t in toks if t not in LEGAL] or toks
    comps = [c for c in (basic(c) for c in addr.split(",")) if c]
    a1 = " ".join(comps)
    atoks = []
    for c in comps:
        ct = c.split()
        if ct and ct[0] in STATE_ABBR and len(ct) <= 2 and all(t.isdigit() and len(t) in (5, 6) for t in ct[1:]):
            ct = STATE_ABBR[ct[0]].split() + ct[1:]
        atoks.extend(ABBR.get(t, t) for t in ct)
    postal, postal_raw = extract_postal(addr)
    return {
        "name_raw": name.strip(),
        "name_n1": n1,
        "name_n2": " ".join(t2),
        "name_n3": " ".join(sorted(t2)),
        "name_tok1": t2[0] if t2 else "",
        "name_tokens": t2,
        "addr_n1": a1,
        "addr_n2": " ".join(sorted(set(atoks))),
        "addr_tokens": atoks,
        "addr_digits": " ".join(sorted({str(int(d)) for d in DIGITS_RE.findall(addr)})),
        "postal": postal,
        "postal_raw": postal_raw,
        "house": extract_house(addr, postal),
        "country": (country or "").strip().casefold(),
    }


def h64(s):
    """Deterministic signed 64-bit hash; 0 is reserved for 'empty key'."""
    if not s:
        return 0
    v = int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                       "little", signed=True)
    return v or 1


def iter_rows(path, io_stats):
    """Yield (row_index, decoded_line) for non-blank data lines. Splits on b'\\n' only."""
    with open(path, "rb") as fb:
        io_stats["header"] = fb.readline().decode("utf-8", "replace").rstrip("\r\n")
        row = 0
        for b in fb:
            if b.endswith(b"\n"):
                b = b[:-1]
            if b.endswith(b"\r"):
                b = b[:-1]
                io_stats["cr_lines"] = io_stats.get("cr_lines", 0) + 1
            if not b.strip():
                io_stats["blank_lines"] = io_stats.get("blank_lines", 0) + 1
                continue
            yield row, b.decode("utf-8", "replace")
            row += 1


def parse_id(eid):
    """S?-<digits> -> int. Non-numeric suffix falls back to a hash (flagged by caller)."""
    try:
        return int(eid[3:]), True
    except ValueError:
        return h64(eid), False


# --------------------------------------------------------------------------------------
# Streaming workers
# --------------------------------------------------------------------------------------

def _reservoir(store, counts, key, item, rng, k=5):
    counts[key] += 1
    lst = store.setdefault(key, [])
    if len(lst) < k:
        lst.append(item)
    else:
        j = rng.randrange(counts[key])
        if j < k:
            lst[j] = item


def profile_source(key, rel):
    path = DATA / rel
    expected_prefix = f"S{key[-1]}-"
    rng = random.Random(SEED)
    cols = {c: array("q") for c in ["id"] + HASH_COLS}
    tok, tokrow = array("q"), array("i")
    name_len, addr_len, name_ntok = array("H"), array("H"), array("B")
    io = {}
    st = Counter()
    missing, placeholders, countries, pat = Counter(), Counter(), Counter(), Counter()
    examples, scripts_name, scripts_addr, legal_tokens = {}, Counter(), Counter(), Counter()
    per_country = defaultdict(Counter)
    country_examples, cex_counts = {}, Counter()
    bad_samples = []

    for row, line in iter_rows(path, io):
        parts = line.split("\t")
        if len(parts) != 4:
            st["bad_field_count"] += 1
            if len(bad_samples) < 5:
                bad_samples.append(line[:300])
            parts = (parts + [""] * 4)[:4]
        eid, name, addr, country = parts
        st["rows"] += 1
        if "�" in line:
            st["replacement_char_rows"] += 1
        eid = eid.strip()
        if not eid.startswith(expected_prefix):
            st["id_prefix_mismatch"] += 1
        idv, numeric = parse_id(eid)
        if not numeric:
            st["id_nonnumeric"] += 1
        elif len(eid) > 4 and eid[3] == "0":
            st["id_leading_zero"] += 1

        cl = country.strip() or "<empty>"
        countries[cl] += 1
        pc = per_country[cl]
        pc["rows"] += 1
        ns, as_, cs = name.strip(), addr.strip(), country.strip()
        if not ns:
            missing["name"] += 1
            pc["name_missing"] += 1
        if not as_:
            missing["address"] += 1
            pc["address_missing"] += 1
        if not cs:
            missing["country"] += 1
        if not ns and not as_:
            missing["name_and_address"] += 1
        if ns.lower() in PLACEHOLDERS:
            placeholders["name:" + ns.lower()] += 1
        if as_.lower() in PLACEHOLDERS:
            placeholders["address:" + as_.lower()] += 1

        rec = normalize_record(name, addr, country)
        cols["id"].append(idv)
        for c in HASH_COLS:
            cols[c].append(h64(rec[c]))
        for t in set(rec["name_tokens"]):
            tok.append(h64(t))
            tokrow.append(row)
        name_len.append(min(len(ns), 65535))
        addr_len.append(min(len(as_), 65535))
        name_ntok.append(min(len(rec["name_tokens"]), 255))

        sn, sa = scripts_of(ns), scripts_of(as_)
        for s in sn:
            scripts_name[s] += 1
        for s in sa:
            scripts_addr[s] += 1
        nonlatin_name = bool(sn - {"LATIN"})
        nonlatin_addr = bool(sa - {"LATIN"})
        if not ns.isascii():
            st["name_nonascii"] += 1
        if not as_.isascii():
            st["address_nonascii"] += 1
        if nonlatin_name:
            st["name_nonlatin"] += 1
            pc["name_nonlatin"] += 1
        if nonlatin_addr:
            st["address_nonlatin"] += 1
            pc["address_nonlatin"] += 1
        if rec["postal"]:
            pc["postal_present"] += 1
        if rec["house"]:
            pc["house_present"] += 1
        n1_tokens = rec["name_n1"].split()
        has_legal = False
        for t in n1_tokens:
            if t in LEGAL:
                legal_tokens[t] += 1
                has_legal = True

        item = [eid, ns[:120], as_[:160], cs]
        flags = []
        if ns:
            if DOMAIN_RE.search(ns):
                flags.append("name_domain_like")
            if not ns[0].isalnum():
                flags.append("name_junk_prefix")
            if name != name.strip():
                flags.append("name_leading_trailing_whitespace")
            if "  " in ns:
                flags.append("name_repeated_spaces")
            if ns == ns.upper() and ns != ns.lower():
                flags.append("name_all_upper")
            if ns == ns.lower() and ns != ns.upper():
                flags.append("name_all_lower")
            if any(ch.isdigit() for ch in ns):
                flags.append("name_has_digits")
            if not any(ch.isalpha() for ch in ns):
                flags.append("name_no_letters")
            if DBA_RE.search(ns):
                flags.append("name_dba_aka")
            if "&" in ns:
                flags.append("name_ampersand")
            if "(" in ns or "[" in ns:
                flags.append("name_parentheses")
            if "@" in ns:
                flags.append("name_email_like")
            if sn and "LATIN" in sn and nonlatin_name:
                flags.append("name_mixed_script")
            if len(rec["name_n1"]) <= 2:
                flags.append("name_very_short")
            if len(ns) > 80:
                flags.append("name_very_long")
            if rec["name_n1"] and rec["name_n1"] == rec["addr_n1"]:
                flags.append("name_equals_address")
            if has_legal:
                flags.append("name_has_legal_token")
        if as_:
            if LANDMARK_RE.search(as_):
                flags.append("addr_landmark")
            if not any(ch.isdigit() for ch in as_):
                flags.append("addr_no_digits")
            if "," not in as_:
                flags.append("addr_single_component")
            if STATEFIRST_RE.match(as_):
                flags.append("addr_starts_with_2letter_code")
            if POBOX_RE.search(as_):
                flags.append("addr_po_box")
            if UNIT_RE.search(as_):
                flags.append("addr_unit_suite_floor")
            if as_ == as_.upper() and as_ != as_.lower():
                flags.append("addr_all_upper")
            if COUNTRYNAME_RE.search(as_):
                flags.append("addr_contains_country_name")
            if rec["postal"]:
                flags.append("addr_has_postal")
            if rec["postal_raw"] and " " in rec["postal_raw"]:
                flags.append("addr_postal_with_space")
        for fl in flags:
            _reservoir(examples, pat, fl, item, rng)
        _reservoir(country_examples, cex_counts, cl, item, rng, k=8)

    CACHE.mkdir(parents=True, exist_ok=True)
    for c, arr in cols.items():
        np.save(CACHE / f"{key}__{c}.npy", np.frombuffer(arr, dtype=np.int64))
    np.save(CACHE / f"{key}__ntok.npy", np.frombuffer(tok, dtype=np.int64))
    np.save(CACHE / f"{key}__ntokrow.npy", np.frombuffer(tokrow, dtype=np.int32))

    def pct(a):
        a = np.frombuffer(a, dtype={"H": np.uint16, "B": np.uint8}[a.typecode])
        if not len(a):
            return {}
        q = np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        return {f"p{p}": float(v) for p, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], q)} | {
            "mean": float(a.mean())}

    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:
        rss = None
    return {
        "key": key, "file": rel, "header": io.get("header"),
        "blank_lines": io.get("blank_lines", 0), "cr_lines": io.get("cr_lines", 0),
        "counters": dict(st), "missing": dict(missing), "placeholders": dict(placeholders),
        "countries": dict(countries.most_common()),
        "per_country": {k: dict(v) for k, v in per_country.items()},
        "patterns": dict(pat.most_common()), "pattern_examples": examples,
        "scripts_name": dict(scripts_name.most_common()),
        "scripts_address": dict(scripts_addr.most_common()),
        "legal_tokens": dict(legal_tokens.most_common(40)),
        "name_len": pct(name_len), "address_len": pct(addr_len), "name_ntokens": pct(name_ntok),
        "country_examples": country_examples, "bad_field_samples": bad_samples,
        "name_token_postings": len(tok), "worker_rss_bytes": rss,
    }


def parse_gt():
    path = DATA / GT_FILE
    io = {}
    st, issues = Counter(), Counter()
    issue_examples = defaultdict(list)
    s1ids, cnt = array("q"), array("i")
    pair_row, pair_t, pair_src = array("i"), array("q"), array("b")
    for row, line in iter_rows(path, io):
        parts = line.split("\t")
        if len(parts) > 2:
            issues["more_than_2_fields"] += 1
        if len(parts) == 1:
            issues["no_tab"] += 1
            parts = parts + [""]
        s1, rest = parts[0].strip(), parts[1]
        st["rows"] += 1
        if not s1.startswith("S1-"):
            issues["s1_bad_prefix"] += 1
        idv, ok = parse_id(s1)
        if not ok:
            issues["s1_nonnumeric"] += 1
        s1ids.append(idv)
        if not rest.strip():
            cnt.append(0)
            continue
        if " " in rest:
            issues["space_in_list"] += 1
        ids = [x.strip() for x in rest.split(",")]
        if any(not x for x in ids):
            issues["empty_token_in_list"] += 1
            ids = [x for x in ids if x]
        if len(ids) != len(set(ids)):
            issues["duplicate_id_in_list"] += 1
            if len(issue_examples["duplicate_id_in_list"]) < 5:
                issue_examples["duplicate_id_in_list"].append(line[:200])
        cnt.append(len(ids))
        for x in ids:
            p = x[:3]
            if p == "S2-":
                src = 2
            elif p == "S3-":
                src = 3
            else:
                issues["self_match" if p == "S1-" else "bad_prefix"] += 1
                continue
            v, ok = parse_id(x)
            if not ok:
                issues["target_nonnumeric"] += 1
            pair_row.append(row)
            pair_t.append(v)
            pair_src.append(src)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "gt__s1.npy", np.frombuffer(s1ids, dtype=np.int64))
    np.save(CACHE / "gt__cnt.npy", np.frombuffer(cnt, dtype=np.int32))
    np.save(CACHE / "gt__pair_row.npy", np.frombuffer(pair_row, dtype=np.int32))
    np.save(CACHE / "gt__pair_t.npy", np.frombuffer(pair_t, dtype=np.int64))
    np.save(CACHE / "gt__pair_src.npy", np.frombuffer(pair_src, dtype=np.int8))
    return {"key": "gt", "file": GT_FILE, "header": io.get("header"),
            "blank_lines": io.get("blank_lines", 0), "cr_lines": io.get("cr_lines", 0),
            "counters": dict(st), "issues": dict(issues),
            "issue_examples": dict(issue_examples), "n_pairs_listed": len(pair_t)}


def run_task(task):
    key, rel, rebuild = task
    stamp_path = CACHE / f"{key}__stamp.json"
    stats_path = CACHE / f"{key}__stats.json"
    fstat = (DATA / rel).stat()
    stamp = {"size": fstat.st_size, "mtime": fstat.st_mtime, "norm_version": NORM_VERSION}
    if not rebuild and stamp_path.exists() and stats_path.exists():
        if json.loads(stamp_path.read_text()) == stamp:
            return key, "cached", 0.0
    t0 = time.time()
    stats = parse_gt() if key == "gt" else profile_source(key, rel)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    stamp_path.write_text(json.dumps(stamp))
    return key, "built", time.time() - t0


# --------------------------------------------------------------------------------------
# In-memory analysis on the hashed arrays
# --------------------------------------------------------------------------------------

M1 = np.uint64(0x9E3779B97F4A7C15)
SH = np.uint64(29)


def load(key, col):
    return np.load(CACHE / f"{key}__{col}.npy", mmap_mode="r")


def comb(*arrs):
    """Combine hash arrays into one key; 0 if any component is empty."""
    n = len(arrs[0])
    out = np.zeros(n, dtype=np.uint64)
    valid = np.ones(n, dtype=bool)
    with np.errstate(over="ignore"):
        for a in arrs:
            a = np.ascontiguousarray(a, dtype=np.int64)
            valid &= a != 0
            out = (out ^ a.view(np.uint64)) * M1
            out ^= out >> SH
    res = out.view(np.int64).copy()
    res[res == 0] = 1
    res[~valid] = 0
    return res


def key_array(split_key, spec):
    if len(spec) == 1:
        return np.asarray(load(split_key, spec[0]))
    return comb(*[load(split_key, c) for c in spec])


def t_key_array(split, spec):
    return np.concatenate([key_array(f"{split}_s2", spec), key_array(f"{split}_s3", spec)])


def uniq_counts(a):
    a = a[a != 0]
    return np.unique(a, return_counts=True)


def candidate_count(u1, c1, uT, cT):
    _, i1, iT = np.intersect1d(u1, uT, assume_unique=True, return_indices=True)
    if not len(i1):
        return {"candidate_pairs": 0, "shared_blocks": 0, "max_block_pairs": 0,
                "max_block_t_size": 0}
    prod = c1[i1].astype(np.int64) * cT[iT].astype(np.int64)
    return {"candidate_pairs": int(prod.sum()), "shared_blocks": int(len(i1)),
            "max_block_pairs": int(prod.max()), "max_block_t_size": int(cT[iT].max())}


class IdIndex:
    def __init__(self, ids):
        ids = np.asarray(ids)
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]

    def rows(self, q):
        q = np.asarray(q)
        if not len(self.sorted):
            return np.full(len(q), -1, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.sorted, q), len(self.sorted) - 1)
        found = self.sorted[pos] == q
        return np.where(found, self.order[pos], -1).astype(np.int64)


def f05_macro(pred, tp, true):
    pred = pred.astype(np.float64)
    tp = tp.astype(np.float64)
    true = true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(pred > 0, tp / pred, 0.0)
        R = np.where(true > 0, tp / true, 0.0)
        F = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    F = np.where(true == 0, (pred == 0).astype(np.float64), F)
    return float(F.mean())


def csr_offsets(rows, n):
    off = np.zeros(n + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.bincount(rows, minlength=n))
    return off


def expand(pid, rows, off, tok):
    starts = off[rows]
    lens = off[rows + 1] - starts
    total = int(lens.sum())
    rep_pid = np.repeat(pid, lens)
    out_start = np.cumsum(lens) - lens
    pos = np.arange(total, dtype=np.int64) - np.repeat(out_start, lens) + np.repeat(starts, lens)
    return rep_pid, tok[pos]


def token_blocking(tok1, row1, n1, tokT, rowT, nT, pair_s1, pair_t, caps, chunk=1_000_000):
    """Blocking on 'shares a name token whose S2+S3 document frequency <= cap'."""
    u1, c1 = np.unique(tok1, return_counts=True)
    uT, cT = np.unique(tokT, return_counts=True)
    _, i1, iT = np.intersect1d(u1, uT, assume_unique=True, return_indices=True)
    dfc = cT[iT].astype(np.int64)
    prod = c1[i1].astype(np.int64) * dfc
    cands = {str(c): int(prod[dfc <= c].sum()) for c in caps}
    del u1, c1, i1, iT, dfc, prod
    off1, offT = csr_offsets(row1, n1), csr_offsets(rowT, nT)
    mindf = np.full(len(pair_s1), np.iinfo(np.int64).max, dtype=np.int64)
    # Process pairs in chunks to bound peak memory (expanded token arrays are ~3x pair count).
    for lo in range(0, len(pair_s1), chunk):
        hi = min(lo + chunk, len(pair_s1))
        pid = np.arange(lo, hi, dtype=np.int64)
        a_pid, a_tok = expand(pid, pair_s1[lo:hi], off1, tok1)
        b_pid, b_tok = expand(pid, pair_t[lo:hi], offT, tokT)
        ka = comb(a_pid + 1, a_tok)
        kb = comb(b_pid + 1, b_tok)
        del b_pid, b_tok
        _, ia, _ = np.intersect1d(ka, kb, return_indices=True)
        del ka, kb
        sp, stok = a_pid[ia], a_tok[ia]
        del a_pid, a_tok
        pos = np.minimum(np.searchsorted(uT, stok), len(uT) - 1)
        np.minimum.at(mindf, sp, cT[pos].astype(np.int64))
    return cands, mindf


def token_candidates_only(tok1, tokT, caps):
    u1, c1 = np.unique(tok1, return_counts=True)
    uT, cT = np.unique(tokT, return_counts=True)
    _, i1, iT = np.intersect1d(u1, uT, assume_unique=True, return_indices=True)
    dfc = cT[iT].astype(np.int64)
    prod = c1[i1].astype(np.int64) * dfc
    return {str(c): int(prod[dfc <= c].sum()) for c in caps}


def fetch_rows(key, rows):
    """Second streaming pass: return {row: [id, name, address, country]} for requested rows."""
    rows = set(int(r) for r in rows)
    out = {}
    if not rows:
        return out
    last = max(rows)
    io = {}
    for row, line in iter_rows(DATA / SOURCE_FILES[key], io):
        if row in rows:
            out[row] = (line.split("\t") + [""] * 4)[:4]
        if row >= last:
            break
    return out


def pair_tags(r1, r2):
    """Describe how a true-match pair differs. r = [id, name, address, country]."""
    f1, f2 = normalize_record(r1[1], r1[2], r1[3]), normalize_record(r2[1], r2[2], r2[3])
    tags = []
    n1r, n2r = r1[1].strip(), r2[1].strip()
    if n1r == n2r:
        tags.append("name_identical")
    elif n1r.casefold() == n2r.casefold():
        tags.append("name_casing_only")
    elif f1["name_n1"] == f2["name_n1"]:
        tags.append("name_punctuation_spacing")
    elif f1["name_n2"] == f2["name_n2"]:
        tags.append("name_legal_suffix")
    elif f1["name_n3"] == f2["name_n3"]:
        tags.append("name_word_order")
    else:
        s1t, s2t = set(f1["name_tokens"]), set(f2["name_tokens"])
        jac = len(s1t & s2t) / max(1, len(s1t | s2t))
        ratio = SequenceMatcher(None, f1["name_n2"], f2["name_n2"]).ratio()
        if s1t and s2t and (s1t <= s2t or s2t <= s1t):
            tags.append("name_token_subset")
        elif ratio >= 0.8:
            tags.append("name_typo_like")
        elif jac == 0:
            tags.append("name_no_shared_token")
        else:
            tags.append("name_other_partial")
    sc1, sc2 = scripts_of(n1r), scripts_of(n2r)
    nl1, nl2 = bool(sc1 - {"LATIN"}), bool(sc2 - {"LATIN"})
    if nl1 or nl2:
        tags.append("name_non_latin")
        if nl1 != nl2:
            tags.append("name_script_differs_transliteration")
    if DOMAIN_RE.search(n1r) or DOMAIN_RE.search(n2r):
        tags.append("name_domain_like")
    if (n1r and not n1r[0].isalnum()) or (n2r and not n2r[0].isalnum()):
        tags.append("name_junk_prefix")
    if not n1r or not n2r:
        tags.append("name_missing_one_side")

    a1r, a2r = r1[2].strip(), r2[2].strip()
    if not a1r and not a2r:
        tags.append("address_both_missing")
    elif not a1r or not a2r:
        tags.append("address_missing_one_side")
    elif a1r == a2r:
        tags.append("address_identical")
    elif f1["addr_n1"] == f2["addr_n1"]:
        tags.append("address_casing_punctuation")
    elif sorted(f1["addr_n1"].split()) == sorted(f2["addr_n1"].split()):
        tags.append("address_reordered")
    elif f1["addr_n2"] == f2["addr_n2"]:
        tags.append("address_abbreviation_or_reorder")
    else:
        s1t, s2t = set(f1["addr_tokens"]), set(f2["addr_tokens"])
        if s1t and s2t and (s1t <= s2t or s2t <= s1t):
            tags.append("address_partial_subset")
        else:
            tags.append("address_other_difference")
    if a1r and a2r:
        p1, p2 = f1["postal"], f2["postal"]
        if p1 and p2:
            if p1 != p2:
                tags.append("postal_mismatch")
            elif f1["postal_raw"] != f2["postal_raw"]:
                tags.append("postal_format_differs")
        elif p1 or p2:
            tags.append("postal_missing_one_side")
        h1, h2 = f1["house"], f2["house"]
        if h1 and h2 and h1 != h2:
            tags.append("house_number_differs")
        m1, m2 = HOUSE_RE.search(a1r), HOUSE_RE.search(a2r)
        if m1 and m2 and m1.group(1) != m2.group(1) and strip_leading_zeros(m1.group(1)) == strip_leading_zeros(m2.group(1)):
            tags.append("house_number_leading_zero_variant")
    if LANDMARK_RE.search(a1r) or LANDMARK_RE.search(a2r):
        tags.append("address_landmark")
    if f1["country"] != f2["country"]:
        tags.append("country_differs")
    return tags, f1, f2


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def analyse(stats, sample_pairs):
    R = {"files": {}, "timing": {}}
    t0 = time.time()
    fetch_req = defaultdict(set)   # key -> rows whose raw text we need
    for k in SOURCE_FILES:
        R["files"][k] = stats[k]
    R["ground_truth_file"] = stats["gt"]

    n = {k: int(stats[k]["counters"]["rows"]) for k in SOURCE_FILES}
    labels = {}
    for k in SOURCE_FILES:
        for c in stats[k]["countries"]:
            lab = c if c != "<empty>" else ""
            labels[h64(lab.strip().casefold())] = lab or "<empty>"

    # ---------------- A. sizes / duplicate IDs -------------------------------------------
    ids = {k: np.asarray(load(k, "id")) for k in SOURCE_FILES}
    dup_ids = {}
    for k, a in ids.items():
        u = np.unique(a)
        dup_ids[k] = int(len(a) - len(u))
    R["dataset_size"] = {
        "rows": n,
        "ground_truth_rows": int(stats["gt"]["counters"]["rows"]),
        "duplicate_entity_ids": dup_ids,
        "file_bytes": {k: (DATA / v).stat().st_size for k, v in SOURCE_FILES.items()}
        | {"gt": (DATA / GT_FILE).stat().st_size},
    }

    # ---------------- B. ground truth -----------------------------------------------------
    gt_s1 = np.asarray(load("gt", "s1"))
    gt_cnt = np.asarray(load("gt", "cnt"))
    pair_row = np.asarray(load("gt", "pair_row")).astype(np.int64)
    pair_t = np.asarray(load("gt", "pair_t"))
    pair_src = np.asarray(load("gt", "pair_src"))
    idx1 = IdIndex(ids["train_s1"])
    idx2 = IdIndex(ids["train_s2"])
    idx3 = IdIndex(ids["train_s3"])
    n1, n2, n3 = n["train_s1"], n["train_s2"], n["train_s3"]
    nT = n2 + n3
    gt_s1row = idx1.rows(gt_s1)
    pair_s1row = gt_s1row[pair_row]
    trow = np.full(len(pair_t), -1, dtype=np.int64)
    m2, m3 = pair_src == 2, pair_src == 3
    trow[m2] = idx2.rows(pair_t[m2])
    r3 = idx3.rows(pair_t[m3])
    trow[m3] = np.where(r3 >= 0, r3 + n2, -1)
    valid = (pair_s1row >= 0) & (trow >= 0)
    del idx1, idx2, idx3, ids  # reloaded for the leakage check; frees ~0.5 GB
    # de-duplicate (s1row, trow) pairs
    pk = pair_s1row * (nT + 1) + trow
    _, first = np.unique(np.where(valid, pk, -1 - np.arange(len(pk))), return_index=True)
    keep = np.zeros(len(pk), dtype=bool)
    keep[first] = True
    keep &= valid
    P_s1 = pair_s1row[keep]
    P_t = trow[keep]
    P_src = pair_src[keep]
    n_pairs = len(P_s1)

    u_gt_s1 = np.unique(gt_s1)
    gt_rows_found = np.unique(gt_s1row[gt_s1row >= 0])
    true_cnt = np.bincount(P_s1, minlength=n1)
    c2 = np.bincount(P_s1[P_src == 2], minlength=n1)
    c3 = np.bincount(P_s1[P_src == 3], minlength=n1)
    dist = np.bincount(true_cnt)
    matched = true_cnt > 0
    tmult = {}
    for s, lo, hi in ((2, 0, n2), (3, n2, nT)):
        tt = P_t[P_src == s]
        u, c = np.unique(tt, return_counts=True)
        tmult[f"S{s}"] = {
            "distinct_targets_in_gt": int(len(u)),
            "targets_matched_to_more_than_one_s1": int((c > 1).sum()),
            "max_s1_per_target": int(c.max()) if len(c) else 0,
            "source_records": int(hi - lo),
            "source_records_never_matched": int((hi - lo) - len(u)),
            "pct_source_records_never_matched": 100.0 * ((hi - lo) - len(u)) / (hi - lo),
        }
    C1 = np.asarray(load("train_s1", "country"))
    CT = t_key_array("train", ("country",))
    pc1, pcT = C1[P_s1], CT[P_t]
    xc = pc1 != pcT
    xc_pairs = Counter()
    if xc.any():
        for a, b in zip(pc1[xc][:200000], pcT[xc][:200000]):
            xc_pairs[f"{labels.get(int(a), a)} -> {labels.get(int(b), b)}"] += 1
    per_country_gt = {}
    for hc, lab in labels.items():
        mask = C1 == hc
        if not mask.any():
            continue
        tc = true_cnt[mask]
        per_country_gt[lab] = {
            "s1_records": int(mask.sum()),
            "pct_singletons": 100.0 * float((tc == 0).mean()),
            "mean_matches": float(tc.mean()),
            "mean_matches_nonsingleton": float(tc[tc > 0].mean()) if (tc > 0).any() else 0.0,
            "mean_s2": float(c2[mask].mean()), "mean_s3": float(c3[mask].mean()),
        }
    R["ground_truth"] = {
        "gt_rows": int(len(gt_s1)),
        "gt_unique_s1": int(len(u_gt_s1)),
        "gt_duplicate_s1_rows": int(len(gt_s1) - len(u_gt_s1)),
        "gt_s1_not_in_train_source1": int((gt_s1row < 0).sum()),
        "train_s1_without_gt_row": int(n1 - len(gt_rows_found)),
        "every_train_s1_has_gt_row": bool(n1 == len(gt_rows_found)),
        "listed_pairs": int(len(pair_t)),
        "pairs_target_not_found_in_source": int((trow < 0).sum()),
        "unique_valid_pairs": int(n_pairs),
        "pairs_s2": int((P_src == 2).sum()), "pairs_s3": int((P_src == 3).sum()),
        "s1_zero_matches": int((~matched).sum()),
        "pct_s1_zero_matches": 100.0 * float((~matched).mean()),
        "match_count_distribution": {str(i): int(v) for i, v in enumerate(dist) if v},
        "s2_count_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(c2)) if v},
        "s3_count_distribution": {str(i): int(v) for i, v in enumerate(np.bincount(c3)) if v},
        "s1_with_s2_match": int((c2 > 0).sum()), "s1_with_s3_match": int((c3 > 0).sum()),
        "s1_with_both_s2_and_s3": int(((c2 > 0) & (c3 > 0)).sum()),
        "s1_only_s2": int(((c2 > 0) & (c3 == 0)).sum()),
        "s1_only_s3": int(((c2 == 0) & (c3 > 0)).sum()),
        "s1_with_2plus_s2": int((c2 >= 2).sum()), "s1_with_2plus_s3": int((c3 >= 2).sum()),
        "max_matches_one_s1": int(true_cnt.max()),
        "mean_matches_all_s1": float(true_cnt.mean()),
        "mean_matches_nonsingleton": float(true_cnt[matched].mean()),
        "median_matches_nonsingleton": float(np.median(true_cnt[matched])),
        "target_multiplicity": tmult,
        "cross_country_pairs": int(xc.sum()),
        "cross_country_breakdown": dict(xc_pairs.most_common(20)),
        "per_country": per_country_gt,
        "predict_all_empty_macro_f05": f05_macro(np.zeros(n1), np.zeros(n1), true_cnt),
    }
    # an example S1 with the maximum number of matches
    mx = int(np.argmax(true_cnt))
    fetch_req["train_s1"].add(mx)
    R["ground_truth"]["_max_example_s1row"] = mx
    R["timing"]["ground_truth"] = time.time() - t0

    # ---------------- C. duplicates of normalised values within each file -----------------
    dup_norm = {}
    for k in SOURCE_FILES:
        dup_norm[k] = {}
        for spec in [("name_n1",), ("name_n2",), ("addr_n1",), ("name_n1", "addr_n1"),
                     ("name_n2", "addr_n2", "country")]:
            a = key_array(k, spec)
            u, c = uniq_counts(a)
            top = np.argsort(c)[-8:][::-1]
            tops = []
            for i in top:
                if c[i] < 2:
                    continue
                r = int(np.flatnonzero(a == u[i])[0])
                fetch_req[k].add(r)
                tops.append({"count": int(c[i]), "row": r})
            dup_norm[k]["+".join(spec)] = {
                "non_empty": int((a != 0).sum()), "distinct": int(len(u)),
                "records_sharing_value_with_another": int(c[c > 1].sum()),
                "pct_records_sharing": 100.0 * float(c[c > 1].sum()) / max(1, len(a)),
                "max_group": int(c.max()) if len(c) else 0, "top": tops,
            }
    R["duplicates_normalized"] = dup_norm
    R["timing"]["duplicates"] = time.time() - t0

    # ---------------- D. blocking keys vs ground truth ------------------------------------
    key_results = {}
    eq_store = {}
    for spec in KEY_SPECS:
        name = "+".join(spec)
        K1 = key_array("train_s1", spec)
        KT = t_key_array("train", spec)
        k1p, kTp = K1[P_s1], KT[P_t]
        both = (k1p != 0) & (kTp != 0)
        eq = both & (k1p == kTp)
        eq_store[name] = np.packbits(eq)
        u1, c1 = uniq_counts(K1)
        uT, cT = uniq_counts(KT)
        cc = candidate_count(u1, c1, uT, cT)
        pos = np.minimum(np.searchsorted(uT, K1), max(len(uT) - 1, 0))
        hit = (K1 != 0) & (uT[pos] == K1) if len(uT) else np.zeros(n1, bool)
        pred = np.where(hit, cT[pos], 0) if len(uT) else np.zeros(n1)
        tp = np.bincount(P_s1[eq], minlength=n1)
        rec_s1 = np.bincount(P_s1[eq], minlength=n1) > 0
        by_country = {}
        for hc, lab in labels.items():
            mk = pc1 == hc
            if mk.any():
                by_country[lab] = 100.0 * float(eq[mk].mean())
        # test side
        K1t = key_array("test_s1", spec)
        KTt = t_key_array("test", spec)
        u1t, c1t = uniq_counts(K1t)
        uTt, cTt = uniq_counts(KTt)
        cct = candidate_count(u1t, c1t, uTt, cTt)
        key_results[name] = {
            "fields": list(spec),
            "s1_coverage_pct": 100.0 * float((K1 != 0).mean()),
            "t_coverage_pct": 100.0 * float((KT != 0).mean()),
            "pairs_both_nonempty": int(both.sum()),
            "agreement_when_both_present_pct": 100.0 * float(eq.sum()) / max(1, int(both.sum())),
            "true_pairs_recovered": int(eq.sum()),
            "pair_recall_pct": 100.0 * float(eq.mean()),
            "pair_recall_s2_pct": 100.0 * float(eq[P_src == 2].mean()),
            "pair_recall_s3_pct": 100.0 * float(eq[P_src == 3].mean()),
            "pair_recall_by_s1_country_pct": by_country,
            "matched_s1_with_any_recovered_pct": 100.0 * float(rec_s1[matched].mean()),
            "train_candidates": cc,
            "pair_precision_pct": 100.0 * float(eq.sum()) / max(1, cc["candidate_pairs"]),
            "macro_f05_if_used_as_matcher": f05_macro(pred, tp, true_cnt),
            "singleton_s1_with_any_candidate_pct": 100.0 * float((pred[~matched] > 0).mean()),
            "test_candidates": cct,
        }
        print(f"    key {name:32s} recall={key_results[name]['pair_recall_pct']:6.2f}% "
              f"prec={key_results[name]['pair_precision_pct']:7.3f}% "
              f"test_cands={cct['candidate_pairs']:,}", flush=True)
        del K1, KT, K1t, KTt
    R["blocking_keys"] = key_results
    R["timing"]["keys"] = time.time() - t0

    # ---------------- D2. name-token blocking --------------------------------------------
    caps = [10, 100, 1000, 10000, 100000, 10 ** 12]
    tok1 = np.asarray(load("train_s1", "ntok"))
    row1 = np.asarray(load("train_s1", "ntokrow")).astype(np.int64)
    tokT = np.concatenate([load("train_s2", "ntok"), load("train_s3", "ntok")])
    rowT = np.concatenate([np.asarray(load("train_s2", "ntokrow")).astype(np.int64),
                           np.asarray(load("train_s3", "ntokrow")).astype(np.int64) + n2])
    tb = {}
    cands, mindf = token_blocking(tok1, row1, n1, tokT, rowT, nT, P_s1, P_t, caps)
    tt1 = np.asarray(load("test_s1", "ntok"))
    ttT = np.concatenate([load("test_s2", "ntok"), load("test_s3", "ntok")])
    tcands = token_candidates_only(tt1, ttT, caps)
    tb["name_token"] = {
        "train_candidates_upper_bound": cands,
        "pair_recall_pct": {str(c): 100.0 * float((mindf <= c).mean()) for c in caps},
        "test_candidates_upper_bound": tcands,
    }
    tok_eq = {"name_token@" + str(c): mindf <= c for c in caps}
    del mindf
    # token + country
    C1c = np.asarray(load("train_s1", "country"))
    tok1c = comb(tok1, C1c[row1])
    tokTc = comb(tokT, CT[rowT])
    cands_c, mindf_c = token_blocking(tok1c, row1, n1, tokTc, rowT, nT, P_s1, P_t, caps)
    t1c = comb(tt1, np.asarray(load("test_s1", "country"))[np.asarray(load("test_s1", "ntokrow"))])
    CTt = t_key_array("test", ("country",))
    n2t = n["test_s2"]
    rowTt = np.concatenate([np.asarray(load("test_s2", "ntokrow")).astype(np.int64),
                            np.asarray(load("test_s3", "ntokrow")).astype(np.int64) + n2t])
    tTc = comb(ttT, CTt[rowTt])
    tcands_c = token_candidates_only(t1c, tTc, caps)
    tb["name_token+country"] = {
        "train_candidates_upper_bound": cands_c,
        "pair_recall_pct": {str(c): 100.0 * float((mindf_c <= c).mean()) for c in caps},
        "test_candidates_upper_bound": tcands_c,
    }
    tb["_note"] = ("Candidate counts are upper bounds (a pair sharing several qualifying tokens "
                   "is counted once per token). cap = max document frequency of the token in "
                   "S2+S3; 1e12 = no cap.")
    R["token_blocking"] = tb
    del tok1, row1, tokT, rowT, tok1c, tokTc, tt1, ttT, t1c, tTc
    R["timing"]["token_blocking"] = time.time() - t0

    # ---------------- D3. unions (recall ceilings) ----------------------------------------
    def union_recall(names, extra=None):
        m = np.zeros(n_pairs, dtype=bool)
        for nm in names:
            m |= np.unpackbits(eq_store[nm], count=n_pairs).astype(bool)
        if extra is not None:
            m |= extra
        s1_any = np.bincount(P_s1[m], minlength=n1) > 0
        return {"keys": names, "pair_recall_pct": 100.0 * float(m.mean()),
                "matched_s1_with_any_recovered_pct": 100.0 * float(s1_any[matched].mean()),
                "test_candidates_upper_bound": int(sum(
                    key_results[x]["test_candidates"]["candidate_pairs"] for x in names))}

    all_exact = [k for k in key_results if k != "country"]
    feasible = [k for k in all_exact
                if key_results[k]["test_candidates"]["candidate_pairs"] <= 2 * 10 ** 8]
    unions = {
        "all_exact_keys_except_country": union_recall(all_exact),
        "exact_keys_with_test_candidates_le_2e8": union_recall(feasible),
    }
    for c in (100, 1000, 10000):
        u = union_recall(feasible, tok_eq["name_token@" + str(c)])
        u["plus"] = f"name_token df<={c}"
        u["test_candidates_upper_bound"] += tb["name_token"]["test_candidates_upper_bound"][str(c)]
        unions[f"feasible_exact_plus_name_token_df_le_{c}"] = u
    mnone = np.zeros(n_pairs, dtype=bool)
    for nm in all_exact:
        mnone |= np.unpackbits(eq_store[nm], count=n_pairs).astype(bool)
    mnone |= tok_eq["name_token@" + str(10 ** 12)]
    unions["_pairs_missed_by_every_exact_key_and_any_shared_name_token_pct"] = \
        100.0 * float((~mnone).mean())
    R["blocking_unions"] = unions
    missed_idx = np.flatnonzero(~mnone)
    del eq_store

    # ---------------- E. noise examples (random sample of true pairs) --------------------
    rng = np.random.default_rng(SEED)
    samp = rng.choice(n_pairs, size=min(sample_pairs, n_pairs), replace=False)
    samp_missed = rng.choice(missed_idx, size=min(200, len(missed_idx)), replace=False) \
        if len(missed_idx) else np.array([], dtype=np.int64)

    def t_loc(t):
        return ("train_s2", int(t)) if t < n2 else ("train_s3", int(t - n2))

    for p in np.concatenate([samp, samp_missed]):
        fetch_req["train_s1"].add(int(P_s1[p]))
        kk, rr = t_loc(P_t[p])
        fetch_req[kk].add(rr)
    # all matches of the max-match S1 (for display)
    mx_t = P_t[P_s1 == mx]
    for t in mx_t:
        kk, rr = t_loc(t)
        fetch_req[kk].add(rr)

    print("  fetching raw text for sampled rows (second streaming pass) ...", flush=True)
    fetched = {k: fetch_rows(k, rows) for k, rows in fetch_req.items()}
    R["timing"]["fetch"] = time.time() - t0

    # resolve top duplicate values to strings
    for k, d in dup_norm.items():
        for spec, v in d.items():
            for t in v["top"]:
                rec = fetched[k].get(t["row"])
                if rec:
                    f = normalize_record(rec[1], rec[2], rec[3])
                    t["example_raw"] = {"name": rec[1][:100], "address": rec[2][:120],
                                        "country": rec[3]}
                    t["normalized"] = {"name_n2": f["name_n2"][:100], "addr_n1": f["addr_n1"][:120]}
    R["ground_truth"]["max_match_example"] = {
        "s1": fetched["train_s1"].get(mx),
        "matches": [fetched[t_loc(t)[0]].get(t_loc(t)[1]) for t in mx_t[:40]],
    }

    tag_counts = Counter()
    by_tag = defaultdict(list)
    name_jacc, addr_jacc = [], []
    shares = Counter()
    for p in samp:
        r1 = fetched["train_s1"][int(P_s1[p])]
        kk, rr = t_loc(P_t[p])
        r2 = fetched[kk][rr]
        tags, f1, f2 = pair_tags(r1, r2)
        for tg in tags:
            tag_counts[tg] += 1
            if len(by_tag[tg]) < 3:
                by_tag[tg].append((r1, r2, f1, f2, tags))
        s1t, s2t = set(f1["name_tokens"]), set(f2["name_tokens"])
        name_jacc.append(len(s1t & s2t) / max(1, len(s1t | s2t)))
        a1t, a2t = set(f1["addr_tokens"]), set(f2["addr_tokens"])
        if a1t and a2t:
            addr_jacc.append(len(a1t & a2t) / len(a1t | a2t))
        shares["pairs"] += 1
        shares["share_any_name_token"] += bool(s1t & s2t)
        shares["share_name_tok1"] += bool(f1["name_tok1"]) and f1["name_tok1"] == f2["name_tok1"]
        shares["share_any_address_token"] += bool(a1t & a2t)
        shares["both_postal_present"] += bool(f1["postal"] and f2["postal"])
        shares["postal_equal"] += bool(f1["postal"]) and f1["postal"] == f2["postal"]
        shares["both_house_present"] += bool(f1["house"] and f2["house"])
        shares["house_equal"] += bool(f1["house"]) and f1["house"] == f2["house"]
        shares["share_any_digit_group"] += bool(set(f1["addr_digits"].split()) &
                                                set(f2["addr_digits"].split()))

    def ex_row(r1, r2, f1, f2, tags):
        return {"s1": {"id": r1[0], "name": r1[1], "address": r1[2], "country": r1[3],
                       "name_n2": f1["name_n2"], "addr_n2": f1["addr_n2"], "postal": f1["postal"],
                       "house": f1["house"]},
                "match": {"id": r2[0], "name": r2[1], "address": r2[2], "country": r2[3],
                          "name_n2": f2["name_n2"], "addr_n2": f2["addr_n2"],
                          "postal": f2["postal"], "house": f2["house"]},
                "differences": tags}

    nj, aj = np.array(name_jacc), np.array(addr_jacc)
    R["noise_sample"] = {
        "sample_size": int(len(samp)),
        "note": "Random sample of true (S1, S2/S3) pairs from the training ground truth.",
        "tag_frequency_pct": {k: 100.0 * v / len(samp) for k, v in tag_counts.most_common()},
        "share_stats_pct": {k: 100.0 * v / shares["pairs"] for k, v in shares.items()
                            if k != "pairs"},
        "name_token_jaccard_quantiles": {f"p{q}": float(np.percentile(nj, q))
                                         for q in (5, 10, 25, 50, 75, 90)},
        "name_token_jaccard_eq0_pct": 100.0 * float((nj == 0).mean()),
        "address_token_jaccard_quantiles": {f"p{q}": float(np.percentile(aj, q))
                                            for q in (5, 10, 25, 50, 75, 90)} if len(aj) else {},
        "examples": {tg: [ex_row(*e) for e in lst] for tg, lst in by_tag.items()},
    }
    missed_examples = []
    for p in samp_missed[:12]:
        r1 = fetched["train_s1"][int(P_s1[p])]
        kk, rr = t_loc(P_t[p])
        r2 = fetched[kk][rr]
        tags, f1, f2 = pair_tags(r1, r2)
        missed_examples.append(ex_row(r1, r2, f1, f2, tags))
    R["noise_sample"]["examples_missed_by_all_exact_keys_and_tokens"] = missed_examples
    R["timing"]["noise"] = time.time() - t0

    # ---------------- F. leakage ---------------------------------------------------------
    leak = {}
    ids = {k: np.asarray(load(k, "id")) for k in SOURCE_FILES}
    for s in ("s1", "s2", "s3"):
        a, b = ids[f"train_{s}"], ids[f"test_{s}"]
        leak[f"{s}_id_overlap_train_test"] = int(len(np.intersect1d(a, b)))
    for s in ("s2", "s3"):
        leak[f"train_s1_vs_test_{s}_numeric_id_overlap_(different_prefix)"] = int(
            len(np.intersect1d(ids["train_s1"], ids[f"test_{s}"])))
    content = {}
    spec = ("name_n1", "addr_n1", "country")
    tr1 = key_array("train_s1", spec)
    te1 = key_array("test_s1", spec)
    trT = t_key_array("train", spec)
    teT = t_key_array("test", spec)
    content["test_s1_name+addr+country_seen_in_train_s1_pct"] = \
        100.0 * float(np.isin(te1[te1 != 0], tr1[tr1 != 0]).mean())
    content["test_s1_name+addr+country_seen_in_train_s2s3_pct"] = \
        100.0 * float(np.isin(te1[te1 != 0], trT[trT != 0]).mean())
    content["test_s2s3_name+addr+country_seen_in_train_s2s3_pct"] = \
        100.0 * float(np.isin(teT[teT != 0], trT[trT != 0]).mean())
    trn = key_array("train_s1", ("name_n2",))
    ten = key_array("test_s1", ("name_n2",))
    content["test_s1_name_n2_seen_in_train_s1_pct"] = \
        100.0 * float(np.isin(ten[ten != 0], trn[trn != 0]).mean())
    leak["content_overlap"] = content
    R["leakage"] = leak

    # ---------------- G. computational ----------------------------------------------------
    R["computational"] = {
        "train_cartesian_s1_x_s2s3": int(n1) * int(nT),
        "test_cartesian_s1_x_s2s3": int(n["test_s1"]) * int(n["test_s2"] + n["test_s3"]),
        "train_records_total": int(n1 + nT),
        "test_records_total": int(sum(n[k] for k in ("test_s1", "test_s2", "test_s3"))),
        "bytes_per_candidate_pair_int32_rows": 8,
        "cache_bytes": int(sum(f.stat().st_size for f in CACHE.glob("*.npy"))),
    }

    # ---------------- H. validation-design facts -----------------------------------------
    multi = []
    for s in (2, 3):
        tt = P_t[P_src == s]
        u, c = np.unique(tt, return_counts=True)
        multi.append(u[c > 1])
    multi_t = np.concatenate(multi) if multi else np.array([], dtype=np.int64)
    involved = np.isin(P_t, multi_t)
    R["validation_facts"] = {
        "s1_entities_linked_through_shared_targets": int(len(np.unique(P_s1[involved]))),
        "train_s1_to_test_s1_ratio": n1 / n["test_s1"],
        "train_t_per_s1": nT / n1,
        "test_t_per_s1": (n["test_s2"] + n["test_s3"]) / n["test_s1"],
    }
    R["timing"]["total_analysis"] = time.time() - t0
    return R


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def fi(x):
    return f"{int(x):,}"


def fp(x, d=2):
    return f"{x:.{d}f}%"


def write_report(R, path):
    L = []
    w = L.append
    ds = R["dataset_size"]
    w("# EDA Report — Business Entity Resolution\n")
    w("Generated by `experiments/eda.py` (read-only, streaming). All numbers are measured; "
      "normalisation rules are simple EDA heuristics (see the key glossary in section D).\n")

    w("## A. Dataset size\n")
    w("| File | Rows | Duplicate IDs | Size (MB) | Countries |")
    w("|---|---:|---:|---:|---|")
    for k in SOURCE_FILES:
        f = R["files"][k]
        cs = ", ".join(f"{c}: {fi(v)} ({100 * v / ds['rows'][k]:.1f}%)"
                       for c, v in f["countries"].items())
        w(f"| {k} | {fi(ds['rows'][k])} | {fi(ds['duplicate_entity_ids'][k])} | "
          f"{ds['file_bytes'][k] / 1e6:.0f} | {cs} |")
    w(f"| ground truth | {fi(ds['ground_truth_rows'])} | — | {ds['file_bytes']['gt'] / 1e6:.0f} | — |\n")
    anomalies = []
    for k in list(SOURCE_FILES) + ["gt"]:
        f = R["files"].get(k) or R["ground_truth_file"]
        c = f.get("counters", {})
        anomalies.append(f"`{k}`: header=`{f.get('header')}`, blank lines={f.get('blank_lines')}, "
                         f"CR line endings={f.get('cr_lines')}, rows with !=4 fields="
                         f"{c.get('bad_field_count', 0)}, rows with U+FFFD={c.get('replacement_char_rows', 0)}, "
                         f"ID prefix mismatches={c.get('id_prefix_mismatch', 0)}, non-numeric IDs="
                         f"{c.get('id_nonnumeric', 0)}, IDs with leading zero={c.get('id_leading_zero', 0)}")
    w("File-format checks:\n")
    for a in anomalies:
        w(f"- {a}")
    gtf = R["ground_truth_file"]
    w(f"- ground-truth list format issues: `{gtf.get('issues')}`\n")

    g = R["ground_truth"]
    w("## B. Ground truth\n")
    w(f"- GT rows: {fi(g['gt_rows'])}; unique S1 in GT: {fi(g['gt_unique_s1'])}; duplicate S1 rows: "
      f"{fi(g['gt_duplicate_s1_rows'])}; GT S1 IDs not in train_source1: {fi(g['gt_s1_not_in_train_source1'])}")
    w(f"- Every train S1 has a GT row: **{g['every_train_s1_has_gt_row']}** "
      f"(train S1 without GT row: {fi(g['train_s1_without_gt_row'])})")
    w(f"- Listed pairs: {fi(g['listed_pairs'])}; pairs whose target ID is missing from train S2/S3: "
      f"{fi(g['pairs_target_not_found_in_source'])}; unique valid pairs: {fi(g['unique_valid_pairs'])} "
      f"(S2: {fi(g['pairs_s2'])}, S3: {fi(g['pairs_s3'])})")
    w(f"- **S1 with zero matches (singletons): {fi(g['s1_zero_matches'])} "
      f"({fp(g['pct_s1_zero_matches'])})** → macro F0.5 of predicting all-empty = "
      f"{g['predict_all_empty_macro_f05']:.4f}")
    w(f"- S1 with ≥1 S2 match: {fi(g['s1_with_s2_match'])}; ≥1 S3 match: {fi(g['s1_with_s3_match'])}; "
      f"**both S2 and S3: {fi(g['s1_with_both_s2_and_s3'])}**; only S2: {fi(g['s1_only_s2'])}; "
      f"only S3: {fi(g['s1_only_s3'])}")
    w(f"- S1 with ≥2 S2 matches: {fi(g['s1_with_2plus_s2'])}; ≥2 S3 matches: {fi(g['s1_with_2plus_s3'])} "
      f"(→ S2/S3 are NOT internally deduplicated)")
    w(f"- Max matches for one S1: **{g['max_matches_one_s1']}**; mean over all S1: "
      f"{g['mean_matches_all_s1']:.3f}; mean/median over non-singletons: "
      f"{g['mean_matches_nonsingleton']:.3f} / {g['median_matches_nonsingleton']:.0f}")
    w(f"- Cross-country true pairs: {fi(g['cross_country_pairs'])} {g['cross_country_breakdown']}\n")
    w("Match-count distribution (matches per S1 → number of S1):\n")
    w("| matches | total | S2 only count | S3 only count |")
    w("|---:|---:|---:|---:|")
    keys = sorted(set(map(int, g["match_count_distribution"])) | set(map(int, g["s2_count_distribution"]))
                  | set(map(int, g["s3_count_distribution"])))
    for kx in keys:
        w(f"| {kx} | {fi(g['match_count_distribution'].get(str(kx), 0))} | "
          f"{fi(g['s2_count_distribution'].get(str(kx), 0))} | {fi(g['s3_count_distribution'].get(str(kx), 0))} |")
    w("")
    w("Target multiplicity (can one S2/S3 record be matched to several S1?):\n")
    w("| Source | distinct targets in GT | targets matched to >1 S1 | max S1 per target | "
      "source records | never matched (distractors) |")
    w("|---|---:|---:|---:|---:|---:|")
    for s, v in g["target_multiplicity"].items():
        w(f"| {s} | {fi(v['distinct_targets_in_gt'])} | {fi(v['targets_matched_to_more_than_one_s1'])} | "
          f"{v['max_s1_per_target']} | {fi(v['source_records'])} | "
          f"{fi(v['source_records_never_matched'])} ({fp(v['pct_source_records_never_matched'])}) |")
    w("")
    w("Per S1 country:\n")
    w("| Country | S1 records | singletons | mean matches | mean (non-singleton) | mean S2 | mean S3 |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for c, v in g["per_country"].items():
        w(f"| {c} | {fi(v['s1_records'])} | {fp(v['pct_singletons'])} | {v['mean_matches']:.3f} | "
          f"{v['mean_matches_nonsingleton']:.3f} | {v['mean_s2']:.3f} | {v['mean_s3']:.3f} |")
    w("")
    mxe = g.get("max_match_example") or {}
    if mxe.get("s1"):
        w(f"Largest cluster example — S1 `{mxe['s1']}`; first matches:\n")
        for m in (mxe.get("matches") or [])[:12]:
            w(f"- `{m}`")
        w("")

    w("## C. Data quality\n")
    w("| File | missing name | missing address | missing country | name non-ASCII | name non-Latin | "
      "address non-ASCII | address non-Latin |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in SOURCE_FILES:
        f = R["files"][k]
        m, c, nr = f["missing"], f["counters"], ds["rows"][k]

        def cell(v):
            return f"{fi(v)} ({100 * v / nr:.2f}%)"
        w(f"| {k} | {cell(m.get('name', 0))} | {cell(m.get('address', 0))} | {cell(m.get('country', 0))} | "
          f"{cell(c.get('name_nonascii', 0))} | {cell(c.get('name_nonlatin', 0))} | "
          f"{cell(c.get('address_nonascii', 0))} | {cell(c.get('address_nonlatin', 0))} |")
    w("")
    w("Placeholder values (e.g. 'nan', '-'):\n")
    for k in SOURCE_FILES:
        w(f"- `{k}`: {R['files'][k]['placeholders'] or 'none'}")
    w("")
    w("Scripts seen in names (records containing letters of that script):\n")
    for k in SOURCE_FILES:
        w(f"- `{k}`: {dict(list(R['files'][k]['scripts_name'].items())[:8])}")
    w("")
    w("Per-country field presence:\n")
    w("| File | Country | rows | address missing | postal found | house no. found | name non-Latin |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for k in SOURCE_FILES:
        for c, v in R["files"][k]["per_country"].items():
            r = v["rows"]
            w(f"| {k} | {c} | {fi(r)} | {fp(100 * v.get('address_missing', 0) / r)} | "
              f"{fp(100 * v.get('postal_present', 0) / r)} | {fp(100 * v.get('house_present', 0) / r)} | "
              f"{fp(100 * v.get('name_nonlatin', 0) / r)} |")
    w("")
    w("Name / address lengths (characters, p50 / p95 / max):\n")
    for k in SOURCE_FILES:
        f = R["files"][k]
        w(f"- `{k}`: name {f['name_len'].get('p50')}/{f['name_len'].get('p95')}/{f['name_len'].get('p100')}, "
          f"address {f['address_len'].get('p50')}/{f['address_len'].get('p95')}/{f['address_len'].get('p100')}, "
          f"name tokens p50 {f['name_ntokens'].get('p50')}")
    w("")
    w("Duplicate normalised values within each file:\n")
    w("| File | key | non-empty | distinct | records sharing value | max group |")
    w("|---|---|---:|---:|---:|---:|")
    for k, d in R["duplicates_normalized"].items():
        for spec, v in d.items():
            w(f"| {k} | {spec} | {fi(v['non_empty'])} | {fi(v['distinct'])} | "
              f"{fi(v['records_sharing_value_with_another'])} ({fp(v['pct_records_sharing'])}) | {fi(v['max_group'])} |")
    w("")
    w("Most frequent normalised names (name_n2) per file:\n")
    for k, d in R["duplicates_normalized"].items():
        tops = d["name_n2"]["top"]
        w(f"- `{k}`: " + "; ".join(f"'{t.get('normalized', {}).get('name_n2', '?')}' ×{fi(t['count'])}"
                                    for t in tops))
    w("")
    w("Most frequent normalised addresses (addr_n1) per file:\n")
    for k, d in R["duplicates_normalized"].items():
        tops = d["addr_n1"]["top"][:5]
        w(f"- `{k}`: " + "; ".join(f"'{t.get('normalized', {}).get('addr_n1', '?')}' ×{fi(t['count'])}"
                                    for t in tops))
    w("")
    w("Unusual name/address patterns (% of rows):\n")
    allp = sorted({p for k in SOURCE_FILES for p in R["files"][k]["patterns"]})
    w("| pattern | " + " | ".join(SOURCE_FILES) + " |")
    w("|---|" + "---:|" * len(SOURCE_FILES))
    for p in allp:
        w(f"| {p} | " + " | ".join(
            fp(100 * R["files"][k]["patterns"].get(p, 0) / ds["rows"][k]) for k in SOURCE_FILES) + " |")
    w("")
    w("Pattern examples (from train_s2 / train_s3):\n")
    for p in allp:
        exs = (R["files"]["train_s2"]["pattern_examples"].get(p, [])[:2]
               + R["files"]["train_s3"]["pattern_examples"].get(p, [])[:1])
        if exs:
            w(f"- **{p}**: " + " · ".join(f"`{e[1]}` | `{e[2]}` ({e[3]})" for e in exs))
    w("")
    w("Legal-suffix tokens found in names (train_s1 / train_s2 / train_s3 top 12):\n")
    for k in ("train_s1", "train_s2", "train_s3", "test_s2"):
        w(f"- `{k}`: {dict(list(R['files'][k]['legal_tokens'].items())[:12])}")
    w("")

    w("## D. Matching structure — exact-key blocking signals\n")
    w("Key glossary:\n")
    for k, v in KEY_DESCRIPTIONS.items():
        w(f"- `{k}`: {v}")
    w("")
    w("*Recall* = % of true (S1,S2/S3) pairs with equal non-empty key. *Precision* = true pairs "
      "recovered / candidate pairs (exact: training GT is complete). *Candidates* = Σ_k n_S1(k)·n_S2S3(k) "
      "(never materialised). *Macro F0.5* = score if every same-key record were predicted as a match "
      "(singletons included).\n")
    w("| key | S1 cov. | recall | recall S2 | recall S3 | agree if both present | matched S1 w/ ≥1 hit | "
      "train cands | precision | macro F0.5 | test cands | max test block (T side) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in R["blocking_keys"].items():
        w(f"| `{k}` | {fp(v['s1_coverage_pct'], 1)} | {fp(v['pair_recall_pct'])} | {fp(v['pair_recall_s2_pct'])} | "
          f"{fp(v['pair_recall_s3_pct'])} | {fp(v['agreement_when_both_present_pct'])} | "
          f"{fp(v['matched_s1_with_any_recovered_pct'])} | {fi(v['train_candidates']['candidate_pairs'])} | "
          f"{fp(v['pair_precision_pct'], 3)} | {v['macro_f05_if_used_as_matcher']:.4f} | "
          f"{fi(v['test_candidates']['candidate_pairs'])} | {fi(v['test_candidates']['max_block_t_size'])} |")
    w("")
    w("Recall by S1 country:\n")
    labs = sorted({c for v in R["blocking_keys"].values() for c in v["pair_recall_by_s1_country_pct"]})
    w("| key | " + " | ".join(labs) + " |")
    w("|---|" + "---:|" * len(labs))
    for k, v in R["blocking_keys"].items():
        w(f"| `{k}` | " + " | ".join(fp(v["pair_recall_by_s1_country_pct"].get(c, 0)) for c in labs) + " |")
    w("")
    w("### Name-token blocking (share ≥1 name_n2 token with S2+S3 document frequency ≤ cap)\n")
    tb = R["token_blocking"]
    w(tb["_note"] + "\n")
    w("| variant | cap | pair recall | train candidates (UB) | test candidates (UB) |")
    w("|---|---:|---:|---:|---:|")
    for var in ("name_token", "name_token+country"):
        v = tb[var]
        for c, r in v["pair_recall_pct"].items():
            capl = "none" if int(c) >= 10 ** 12 else fi(c)
            w(f"| {var} | {capl} | {fp(r)} | {fi(v['train_candidates_upper_bound'][c])} | "
              f"{fi(v['test_candidates_upper_bound'][c])} |")
    w("")
    w("### Unions (recall ceilings)\n")
    w("| union | pair recall | matched S1 with ≥1 hit | test candidates (upper bound, sum) |")
    w("|---|---:|---:|---:|")
    for k, v in R["blocking_unions"].items():
        if k.startswith("_"):
            continue
        w(f"| {k} ({len(v['keys'])} keys{', + ' + v['plus'] if 'plus' in v else ''}) | "
          f"{fp(v['pair_recall_pct'])} | {fp(v['matched_s1_with_any_recovered_pct'])} | "
          f"{fi(v['test_candidates_upper_bound'])} |")
    w(f"\nTrue pairs missed by every exact key AND sharing no name token at all: "
      f"{fp(R['blocking_unions']['_pairs_missed_by_every_exact_key_and_any_shared_name_token_pct'])}\n")

    ns = R["noise_sample"]
    w(f"## E. Noise examples (random sample of {fi(ns['sample_size'])} true pairs)\n")
    w("Difference-tag frequency (a pair can carry several tags; name/address primary tags are exclusive):\n")
    w("| tag | % of sampled true pairs |")
    w("|---|---:|")
    for k, v in ns["tag_frequency_pct"].items():
        w(f"| {k} | {fp(v)} |")
    w("")
    w("Shared-signal rates on sampled true pairs:\n")
    for k, v in ns["share_stats_pct"].items():
        w(f"- {k}: {fp(v)}")
    w(f"- name token Jaccard quantiles: {ns['name_token_jaccard_quantiles']} (Jaccard = 0 for "
      f"{fp(ns['name_token_jaccard_eq0_pct'])})")
    w(f"- address token Jaccard quantiles: {ns['address_token_jaccard_quantiles']}\n")
    order = ["name_casing_only", "name_punctuation_spacing", "name_legal_suffix", "name_word_order",
             "name_typo_like", "name_token_subset", "name_no_shared_token", "name_non_latin",
             "name_script_differs_transliteration", "name_domain_like", "name_junk_prefix",
             "address_reordered", "address_abbreviation_or_reorder", "address_partial_subset",
             "address_missing_one_side", "postal_format_differs", "postal_missing_one_side",
             "postal_mismatch", "house_number_differs", "house_number_leading_zero_variant", "address_landmark", "country_differs"]
    for tg in order + [t for t in ns["examples"] if t not in order]:
        exs = ns["examples"].get(tg)
        if not exs:
            continue
        w(f"### {tg}\n")
        for e in exs[:2]:
            s, m = e["s1"], e["match"]
            w(f"- S1 `{s['id']}`: **{s['name']}** | {s['address']} | {s['country']}  ")
            w(f"  → normalised: `{s['name_n2']}` | `{s['addr_n2']}` | postal=`{s['postal']}` house=`{s['house']}`  ")
            w(f"  {m['id'][:2]} `{m['id']}`: **{m['name']}** | {m['address']} | {m['country']}  ")
            w(f"  → normalised: `{m['name_n2']}` | `{m['addr_n2']}` | postal=`{m['postal']}` house=`{m['house']}`  ")
            w(f"  differences: {', '.join(e['differences'])}")
        w("")
    w("### True pairs missed by every exact key and sharing no name token\n")
    for e in ns["examples_missed_by_all_exact_keys_and_tokens"][:10]:
        s, m = e["s1"], e["match"]
        w(f"- **{s['name']}** | {s['address']} ({s['country']})  ⟷  **{m['name']}** | {m['address']} "
          f"({m['country']}) — {', '.join(e['differences'])}")
    w("")
    w("### Country examples in test (incl. unseen France)\n")
    for k in ("test_s1", "test_s2", "test_s3"):
        for c, exs in R["files"][k]["country_examples"].items():
            w(f"- `{k}` {c}: " + " · ".join(f"`{e[1]}` | `{e[2]}`" for e in exs[:3]))
    w("")

    lk = R["leakage"]
    w("## F. Train/test leakage\n")
    for k, v in lk.items():
        if k != "content_overlap":
            w(f"- {k}: {fi(v)}")
    for k, v in lk["content_overlap"].items():
        w(f"- {k}: {fp(v)}")
    w("")

    cp = R["computational"]
    w("## G. Computational\n")
    w(f"- Train records: {fi(cp['train_records_total'])}; test records: {fi(cp['test_records_total'])}")
    w(f"- Full Cartesian S1×(S2+S3): train {cp['train_cartesian_s1_x_s2s3']:.3e}, "
      f"test {cp['test_cartesian_s1_x_s2s3']:.3e} pairs — infeasible (at 8 bytes/pair ≈ "
      f"{cp['test_cartesian_s1_x_s2s3'] * 8 / 1e12:.0f} TB).")
    w(f"- EDA hash cache on disk: {cp['cache_bytes'] / 1e9:.2f} GB (8 bytes per key per record).")
    w("- Worker peak RSS while streaming (bytes): " + ", ".join(
        f"{k}={R['files'][k].get('worker_rss_bytes') or 0:,}" for k in SOURCE_FILES))
    w("")
    vf = R["validation_facts"]
    w("## H. Validation-design facts\n")
    w(f"- S1 entities linked to another S1 via a shared S2/S3 target: {fi(vf['s1_entities_linked_through_shared_targets'])}")
    w(f"- S2+S3 records per S1: train {vf['train_t_per_s1']:.3f}, test {vf['test_t_per_s1']:.3f}")
    w(f"- train S1 / test S1 = {vf['train_s1_to_test_s1_ratio']:.3f}")
    w("")
    w(f"Timing (s): {R['timing']}")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true", help="ignore the hash cache")
    ap.add_argument("--sample-pairs", type=int, default=30000)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    tasks = [(k, rel, args.rebuild) for k, rel in SOURCE_FILES.items()] + [("gt", GT_FILE, args.rebuild)]
    tasks.sort(key=lambda t: -(DATA / t[1]).stat().st_size)
    t0 = time.time()
    print(f"[1/3] streaming {len(tasks)} files with {args.workers} workers ...", flush=True)
    with Pool(args.workers) as pool:
        for key, status, secs in pool.imap_unordered(run_task, tasks):
            print(f"  {key}: {status} ({secs:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)
    stats = {k: json.loads((CACHE / f"{k}__stats.json").read_text(encoding="utf-8"))
             for k in list(SOURCE_FILES) + ["gt"]}
    print("[2/3] analysing hashed arrays ...", flush=True)
    R = analyse(stats, args.sample_pairs)
    R["timing"]["streaming_wall"] = None
    R["timing"]["script_wall"] = time.time() - t0
    R = to_jsonable(R)
    print("[3/3] writing outputs ...", flush=True)
    (OUT / "eda_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(R, OUT / "EDA_REPORT.md")
    print(f"done in {time.time() - t0:.0f}s -> experiments/eda_results.json, experiments/EDA_REPORT.md")


if __name__ == "__main__":
    main()
