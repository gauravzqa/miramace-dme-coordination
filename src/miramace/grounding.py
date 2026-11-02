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
      | \d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?                           # 11/5
      | today|tomorrow|tonight
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_MONEY_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?")
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
}
_ABBREV_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|Ave|Rd|N|S|E|W)\.")
_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+|\s+[—–-]\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def _norm_date(s: str) -> str:
    s = s.lower().replace(".", "").replace(",", " ")
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\b(" + "|".join(WEEKDAYS) + r")s\b", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _money_val(s: str) -> str:
    v = s.replace("$", "").replace(",", "").strip().rstrip(".")
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


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
