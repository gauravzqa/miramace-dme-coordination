"""B4 Patient side — grounding check: every date, $ amount, percentage and proper name in a message to
Eleanor must come from the case snapshot (or her profile). Returns violations; empty list = grounded.
"""

from __future__ import annotations

import re

from miramace.models import PatientProfile, Snapshot
from miramace.snapshot import snapshot_text

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_ABBR = r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
_UNAMBIGUOUS_MONTHS = r"january|february|april|june|july|september|october|november|december"

_DATE_RE = re.compile(
    rf"""\b(?:
        (?:{"|".join(WEEKDAYS)})s?                                   # weekday
      | (?:{"|".join(MONTHS)}|{_MONTH_ABBR})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b   # month + day
      | (?:{_UNAMBIGUOUS_MONTHS})                                    # month alone (not May/March)
      | \d{{4}}-\d{{1,2}}-\d{{1,2}}                                     # 2026-11-05
      | \d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?                           # 11/5
      | today|tomorrow|tonight
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_MONEY_RE = re.compile(
    r"\$\s?[\d,]+(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:dollars?|bucks|usd)\b", re.IGNORECASE
)
_NUMBER_WORDS = {
    "a": "1",
    "an": "1",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "a couple of": "2",
    "a couple": "2",
    "a few": "few",
    "few": "few",
    "several": "several",
}
_NUM_ALT = "|".join(
    sorted((re.escape(k) for k in _NUMBER_WORDS if k not in ("a", "an")), key=len, reverse=True)
)
# relative durations / lead times ("within 2 days", "in three weeks", "a couple of business days")
_DURATION_RE = re.compile(
    rf"\b(\d+|{_NUM_ALT}|a|an)\s+(?:more\s+)?(?:business\s+|working\s+)?"
    r"(hours?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b)", re.IGNORECASE)

STOPLIST = {
    "i",
    "medicare",
    "mira",
    "mace",
    "ok",
    "okay",
    "eleanor",
    "dr",
    "hi",
    "hello",
    "am",
    "pm",
    "part",
    "b",
    "original",
    "medigap",
    "zip",
    "thanks",
    "thank",
    "please",
    "yes",
    "no",
    "mr",
    "mrs",
    "ms",
    "dme",
    "us",
    "tv",
    # domain vocabulary that is not a fact (supplier / person names must still come from the snapshot)
    "care",
    "advocate",
    "nurse",
    "team",
    "k0001",
    "hcpcs",
    "lcd",
    "swo",
    "npi",
    "mbi",
    "coinsurance",
    "deductible",
    "wheelchair",
}
_ABBREV_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|Ave|Rd|N|S|E|W)\.")
_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+|\s+[—–-]\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")


def _norm_date(s: str) -> str:
    iso = re.fullmatch(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*", s)
    if iso:
        return f"{int(iso.group(2))}/{int(iso.group(3))}"
    num = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\s*", s)
    if num:
        return f"{int(num.group(1))}/{int(num.group(2))}"
    s = s.lower().replace(".", "").replace(",", " ")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\b(" + "|".join(WEEKDAYS) + r")s\b", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _money_val(s: str) -> str:
    v = re.sub(r"(?i)\s*(dollars?|bucks|usd)$", "", s.strip())
    v = v.replace("$", "").replace(",", "").strip().rstrip(".")
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def _duration_val(m: re.Match[str]) -> str:
    n = m.group(1).lower()
    n = _NUMBER_WORDS.get(n, n)
    unit = m.group(2).lower().rstrip("s")
    return f"{n} {unit}"


def _allowed_durations(snapshot: Snapshot, profile: PatientProfile | None) -> set[str]:
    corpus = [snapshot_text(snapshot), *snapshot.allowed_amounts]
    if profile is not None:
        corpus += list(profile.prior_concerns)
    return {_duration_val(m) for chunk in corpus for m in _DURATION_RE.finditer(chunk)}


def _pct_val(s: str) -> str:
    return re.sub(r"\s?(%|percent)$", "", s.strip().lower())


def _word_core(w: str) -> str:
    w = w.replace("’", "'")
    w = re.sub(r"'(s|ll|m|re|ve|d|t)$", "", w, flags=re.IGNORECASE)
    return w.strip("'-")


def _allowed_tokens(snapshot: Snapshot, profile: PatientProfile | None) -> set[str]:
    corpus = [snapshot_text(snapshot), *snapshot.allowed_names]
    if profile is not None:
        corpus += [profile.caregiver_name or "", profile.home_access or "", profile.hearing_or_language or ""]
        corpus += list(profile.prior_concerns)
    toks: set[str] = set()
    for chunk in corpus:
        for w in _WORD_RE.findall(chunk):
            core = _word_core(w).lower()
            toks.add(core)
            toks.update(p for p in core.split("-") if p)
    return toks


def grounding_check(text: str, snapshot: Snapshot, profile: PatientProfile | None = None) -> list[str]:
    """Return human-readable violations (each names the offending fragment). Empty = grounded."""
    violations: list[str] = []

    def flag(kind: str, frag: str) -> None:
        v = f"{kind}: {frag}"
        if v not in violations:
            violations.append(v)

    # dates
    allowed_dates = [_norm_date(d) for d in snapshot.allowed_dates]
    for m in _DATE_RE.finditer(text):
        frag = _norm_date(m.group(0))
        ok = frag in allowed_dates or any(re.search(rf"\b{re.escape(frag)}\b", a) for a in allowed_dates)
        if not ok:
            flag("date", m.group(0))

    # money
    allowed_money = {_money_val(a) for a in snapshot.allowed_amounts if a.strip().startswith("$")}
    for m in _MONEY_RE.finditer(text):
        if _money_val(m.group(0)) not in allowed_money:
            flag("amount", m.group(0).rstrip("."))

    # relative durations / lead times ("within 2 days") must be stated in the snapshot
    allowed_dur = _allowed_durations(snapshot, profile)
    for m in _DURATION_RE.finditer(text):
        if _duration_val(m) not in allowed_dur:
            flag("duration", m.group(0))

    # percentages
    allowed_pct = {_pct_val(a) for a in snapshot.allowed_amounts if a.strip().endswith(("%", "percent"))}
    for m in _PCT_RE.finditer(text):
        if _pct_val(m.group(0)) not in allowed_pct:
            flag("percent", m.group(0))

    # capitalised, non-sentence-initial tokens = probable names
    allowed = _allowed_tokens(snapshot, profile)
    date_words = set(WEEKDAYS) | set(MONTHS)
    body = _ABBREV_RE.sub(r"\1", text)
    for sent in _SENT_SPLIT.split(body):
        words = _WORD_RE.findall(sent)
        for w in words[1:]:
            core = _word_core(w)
            if not core or not core[0].isupper():
                continue
            low = core.lower()
            if low in STOPLIST or low in allowed or low in date_words or low.rstrip("s") in date_words:
                continue
            if "-" in low and all(p in allowed or p in STOPLIST for p in low.split("-") if p):
                continue
            flag("name", core)
    return violations
