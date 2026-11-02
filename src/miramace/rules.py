"""B2 — pure domain rules. No I/O, no clock reads, no LLM. Depends only on frozen models.

Medicare facts encoded here are constants (CostInfo defaults); never invent amounts.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from miramace.geo import distance_mi
from miramace.models import (
    LCD_CRITERION_BY_LLM_ID,
    SWO,
    TARGET_HCPCS,
    AssignmentStatus,
    CaseState,
    Claim,
    CostInfo,
    LCDChecklist,
    LCDCriterion,
    LCDResult,
    PatientStatus,
    Supplier,
    SupplierFacts,
)

FRESHNESS = timedelta(hours=24)  # supplier verification must be this fresh to schedule (invariant 1)

ELIGIBILITY_FIELDS: tuple[str, ...] = (
    "taking_new_medicare",
    "k0001_in_stock",
    "delivers_to_zip",
    "accepts_assignment_all_months",
)

# The SWO elements we require (CMS SWO elements as modelled on `SWO`), then signature.
SWO_ELEMENTS: tuple[str, ...] = (
    "beneficiary_name",
    "mbi",
    "order_date",
    "item_description",
    "hcpcs",
    "quantity",
    "practitioner_name",
    "practitioner_npi",
)

# ------------------------------------------------------------------------------------------ suppliers


def rank_suppliers(suppliers: list[Supplier], patient_zip: str) -> list[tuple[str, float]]:
    """[(sid, miles)] ascending by distance from the patient's ZIP; ties broken by supplier name."""
    scored = [(distance_mi(patient_zip, s.zip), s.name, s.sid) for s in suppliers]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [(sid, miles) for miles, _name, sid in scored]


def supplier_eligible(f: SupplierFacts | None) -> tuple[bool, str | None]:
    """All four eligibility booleans must be True.

    A definite False wins over an unknown: reason "not:<field>". Otherwise a None gives "unknown:<field>"
    (caller policy: one re-screen, then DISQUALIFIED).
    """
    if f is None:
        return False, f"unknown:{ELIGIBILITY_FIELDS[0]}"
    for field in ELIGIBILITY_FIELDS:
        if getattr(f, field) is False:
            return False, f"not:{field}"
    for field in ELIGIBILITY_FIELDS:
        if getattr(f, field) is None:
            return False, f"unknown:{field}"
    return True, None


def facts_consistent(f: SupplierFacts) -> list[str]:
    """Internal contradictions in what a supplier told us (each issue is a short string)."""
    issues: list[str] = []
    days = f.earliest_delivery_days
    if days is not None and days < 0:
        issues.append(f"earliest_delivery_days negative ({days})")
    if f.k0001_in_stock is False and days is not None and days <= 3:
        issues.append(f"out of stock but claims delivery in {days} day(s)")
    if f.delivers_to_zip is False and days is not None:
        issues.append("does not deliver to ZIP but quoted a delivery time")
    for field, quote in f.evidence.items():
        if not quote or not quote.strip():
            issues.append(f"empty evidence quote for {field}")
    return issues


def assignment_ok(sid: str | None, state: CaseState) -> bool:
    """True only when assignment for *all rental months* was confirmed for this supplier."""
    if not sid:
        return False
    return state.assignment_by_sid.get(sid) == AssignmentStatus.CONFIRMED_ALL_MONTHS


# ------------------------------------------------------------------------------------------ order docs


def swo_completeness(swo: SWO | None) -> list[str]:
    """Missing SWO element names (field names), plus 'signed' / 'signature_date'."""
    if swo is None:
        return [*SWO_ELEMENTS, "signed", "signature_date"]
    missing: list[str] = []
    for field in SWO_ELEMENTS:
        v = getattr(swo, field)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(field)
        elif field == "quantity" and v < 1:
            missing.append(field)
    if not swo.signed:
        missing.append("signed")
    if swo.signature_date is None:
        missing.append("signature_date")
    return missing


def check_code(hcpcs: str | None) -> str:
    """'OK' for K0001, 'MISSING' when blank, 'DRIFT' for anything else (e.g. K0002–K0007)."""
    if hcpcs is None or not str(hcpcs).strip():
        return "MISSING"
    return "OK" if str(hcpcs).strip().upper() == TARGET_HCPCS else "DRIFT"


_LCD_TRUE = {"met", "true", "yes", "y", "documented", "present", "confirmed"}
_LCD_FALSE = {"not_met", "not met", "false", "no", "n", "absent"}
_LCD_UNDOC = {"not_documented", "not documented", "unknown", "unclear", "none", ""}


def _criterion(field: str) -> LCDCriterion | None:
    key = field.strip()
    if key in LCD_CRITERION_BY_LLM_ID:
        return LCD_CRITERION_BY_LLM_ID[key]
    try:
        return LCDCriterion(key.upper())
    except ValueError:
        return None


def _lcd_value(value: str | None) -> LCDResult:
    v = (value or "").strip().lower()
    if v in _LCD_TRUE:
        return "met"
    if v in _LCD_FALSE:
        return "not_met"
    if v in _LCD_UNDOC:
        return "not_documented"
    b = parse_bool(v)
    if b is True:
        return "met"
    if b is False:
        return "not_met"
    return "not_documented"  # conservative: free text we cannot read as true is not "met"


_RANK: dict[str, int] = {"not_documented": 0, "not_met": 1, "met": 2}


def lcd_evaluate(claims: list[Claim]) -> LCDChecklist:
    """Checklist over all 5 criteria. 'met' only if status=='stated', value truthy AND evidence_quote present.

    Claim.field may be an LCDCriterion name or an LLM criterion id. Multiple claims for one criterion
    (note + addendum): the strongest result wins (met > not_met > not_documented).
    """
    results: dict[LCDCriterion, LCDResult] = {c: "not_documented" for c in LCDCriterion}
    evidence: dict[LCDCriterion, str] = {}
    for cl in claims:
        crit = _criterion(cl.field)
        if crit is None:
            continue
        quote = (cl.evidence_quote or "").strip()
        if cl.status != "stated" or not quote:
            res: LCDResult = "not_documented"
        else:
            res = _lcd_value(cl.value)
        if _RANK[res] > _RANK[results[crit]]:
            results[crit] = res
            if quote and res != "not_documented":
                evidence[crit] = quote
    return LCDChecklist(results=results, evidence=evidence)


_LCD_WORDING: dict[LCDCriterion, str] = {
    LCDCriterion.MRADL_LIMITATION: (
        "a statement that her mobility limitation keeps her from doing everyday activities at home "
        "(like getting to the bathroom, dressing or the kitchen) safely or in a reasonable time"
    ),
    LCDCriterion.CANE_WALKER_INSUFFICIENT: (
        "a statement that a cane or walker is not enough to solve that mobility limitation"
    ),
    LCDCriterion.HOME_ACCOMMODATES: (
        "a statement that her home has enough room (doorways, hallways, floor surfaces) to use a manual "
        "wheelchair"
    ),
    LCDCriterion.WILL_USE_IMPROVES: (
        "a statement that using the wheelchair will meaningfully improve her ability to do those activities "
        "and that she will use it regularly at home"
    ),
    LCDCriterion.SELF_PROPEL_OR_CAREGIVER: (
        "a statement that she can propel the wheelchair herself, or has a caregiver who is available and "
        "willing to push it"
    ),
}


def lcd_missing_wording(criterion: LCDCriterion | str) -> str:
    """Plain-English request text for one missing LCD criterion (used when asking for an addendum)."""
    crit = criterion if isinstance(criterion, LCDCriterion) else _criterion(str(criterion))
    if crit is None:
        raise ValueError(f"unknown LCD criterion: {criterion!r}")
    return _LCD_WORDING[crit]


# ------------------------------------------------------------------------------------------ scheduling


def can_schedule(state: CaseState, now: datetime) -> tuple[bool, list[str]]:
    """Invariant-1 predicate: may a delivery be scheduled right now? Returns (ok, reasons-it-cannot)."""
    reasons: list[str] = []
    sid = state.committed_sid
    if sid is None:
        reasons.append("no committed supplier")
    if state.order_received_by is None or state.order_received_by != sid:
        reasons.append("order not received by committed supplier")
    swo = state.swo
    if swo is None:
        reasons.append("no signed SWO")
    else:
        if check_code(swo.hcpcs) != "OK":
            reasons.append(f"SWO code is {swo.hcpcs!r}, not {TARGET_HCPCS}")
        missing = swo_completeness(swo)
        if missing:
            reasons.append("SWO incomplete: " + ", ".join(missing))
    if state.lcd is None or not state.lcd.all_met:
        missing_lcd = state.lcd.missing if state.lcd else list(LCDCriterion)
        reasons.append("LCD criteria not met: " + ", ".join(m.value for m in missing_lcd))
    if not assignment_ok(sid, state):
        reasons.append("assignment not confirmed for committed supplier")
    rec = state.suppliers.get(sid) if sid else None
    if rec is None or rec.verified_at is None:
        reasons.append("committed supplier never verified")
    elif now - rec.verified_at > FRESHNESS or rec.verified_at > now:
        reasons.append("supplier verification older than 24h")
    if state.patient.state != PatientStatus.COST_ACCEPTED:
        reasons.append("patient has not accepted cost")
    return (not reasons, reasons)


def cost_info() -> CostInfo:
    """The only allowed cost facts (2026 Part B deductible, 20% coinsurance, 13-month capped rental)."""
    return CostInfo()


# ------------------------------------------------------------------------------------------ parsing

_NEG_OK = re.compile(r"\b(no problem|not a problem|no worries)\b")
_NEG = re.compile(
    r"\b(no|nope|nah|not|never|none|negative|false|unfortunately|cannot|can't|cant|won't|wont|"
    r"don't|dont|doesn't|doesnt|isn't|isnt|aren't|arent|do not|does not|we don't)\b"
)
_POS = re.compile(
    r"\b(yes|yep|yeah|yup|ya|sure|correct|absolutely|definitely|certainly|of course|affirmative|true|"
    r"we do|we are|we can|we have|we will|that's right|thats right|indeed|y)\b"
)


def parse_bool(text: str | None) -> bool | None:
    """Yes/no answer → bool; None when it can't be read. Negation wins ("yes, but not at the moment")."""
    if text is None:
        return None
    t = text.strip().lower().replace("’", "'")
    if not t:
        return None
    t_wo_ok = _NEG_OK.sub(" ", t)
    if _NEG.search(t_wo_ok):
        return False
    if _POS.search(t) or t_wo_ok != t:
        return True
    return None
