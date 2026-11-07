"""Text normalisation shared by the EDA, blocking and (later) matching stages.

The block between the markers below is copied VERBATIM from ``experiments/eda.py`` (NORM_VERSION 2),
which produced the EDA hash cache (``experiments/_eda_cache``). ``src/tests/test_normalize.py``
asserts that keys computed here are bit-identical to that cache, so blocking and EDA statistics use
exactly the same rules. Do not edit the verbatim block without bumping NORM_VERSION and rebuilding.

Blocking-specific tokenisers (``blocking_tokens``, ``record_flags``) are appended after the block.
"""

import hashlib
import re
import unicodedata

NORM_VERSION = 2

# ===== BEGIN verbatim copy of experiments/eda.py (normalisation) =====
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

# ===== END verbatim copy =====


# ---------------------------------------------------------------------------------------------
# Blocking-specific additions (not part of the EDA copy)
# ---------------------------------------------------------------------------------------------

# Exact-key columns, identical to experiments/eda.py HASH_COLS (same order, same hashing).
KEY_COLS = [
    "name_raw", "name_n1", "name_n2", "name_n3", "name_tok1",
    "addr_n1", "addr_n2", "addr_digits", "postal", "house", "country",
]

# Tokens dropped from domain-like names only ("desertbsp.com" -> core "desertbsp").
TLD_TOKENS = {"com", "net", "org", "in", "co", "io", "biz", "info", "fr", "us", "www",
              "http", "https", "shop", "store", "online"}

FLAG_NAME_MISSING = 1
FLAG_ADDR_MISSING = 2
FLAG_COUNTRY_MISSING = 4
FLAG_NAME_NONLATIN = 8
FLAG_NAME_DOMAIN = 16
FLAG_NAME_JUNK_PREFIX = 32
FLAG_POSTAL_PRESENT = 64
FLAG_ADDR_LANDMARK = 128


def blocking_tokens(rec, raw_name):
    """Return ``(name_tokens, address_tokens)`` as sets of namespaced strings.

    name:    ``w:<word>`` for each name_n2 word (legal suffixes removed, initials joined) and
             ``g:<trigram>`` over ``#<words joined without spaces>#`` — robust to typos, word splits /
             concatenation (domain-style names), junk prefixes and word order.
    address: ``a:<token>`` for alphabetic address tokens (street/state abbreviations expanded as in
             addr_n2), ``d:<int>`` for digit groups (leading zeros removed), ``h:<house number>``.
    """
    words = rec["name_tokens"]
    if raw_name and DOMAIN_RE.search(raw_name):
        words = [w for w in words if w not in TLD_TOKENS] or words
    name = {"w:" + w for w in words}
    core = "".join(words)
    if core:
        s = f"#{core}#"
        name.update("g:" + s[i:i + 3] for i in range(len(s) - 2))
    addr = set()
    for t in rec["addr_tokens"]:
        if t.isdigit():
            addr.add("d:" + str(int(t)))
        else:
            addr.add("a:" + t)
    if rec["house"]:
        addr.add("h:" + rec["house"])
    return name, addr


def record_flags(name, addr, country, rec):
    """Bitmask of record-level edge-case flags (see FLAG_* constants)."""
    ns, as_ = (name or "").strip(), (addr or "").strip()
    f = 0
    if not ns:
        f |= FLAG_NAME_MISSING
    if not as_:
        f |= FLAG_ADDR_MISSING
    if not (country or "").strip():
        f |= FLAG_COUNTRY_MISSING
    if scripts_of(ns) - {"LATIN"}:
        f |= FLAG_NAME_NONLATIN
    if ns and DOMAIN_RE.search(ns):
        f |= FLAG_NAME_DOMAIN
    if ns and not ns[0].isalnum():
        f |= FLAG_NAME_JUNK_PREFIX
    if rec["postal"]:
        f |= FLAG_POSTAL_PRESENT
    if as_ and LANDMARK_RE.search(as_):
        f |= FLAG_ADDR_LANDMARK
    return f
