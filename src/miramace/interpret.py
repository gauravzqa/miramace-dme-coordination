"""B6 Engine — interpret: (task | doc, CallResult | claims, state, now) -> domain Events. Pure.

The AI layers produce Claims (field, status, value, verbatim quote). Here CODE decides what they mean:
- every stated claim's quote is re-verified against the counterparty's words (call: one non-injection
  counterparty utterance; doc: the document text). Failures -> EXTRACTION_FAILED + treated as unknown;
- values are parsed in code: booleans via rules.parse_bool, relative dates ("Thursday", "by Friday",
  "tomorrow", "in 3 days", "10am") via `parse_when` against the virtual clock, ints via `parse_int`;
- rules decide outcomes (rules.supplier_eligible, rules.lcd_evaluate, rules.swo_completeness).
Every event is sourced TRANSCRIPT(ref=transcript_id) or INBOUND_DOC(ref=doc_id).

Call-intent conventions (slot ids per intent are defined in executor.SLOTS):
  pcp request_order          -> ORDER_REQUESTED(v) + COMMITMENT_MADE(pcp, promised_action, promised_by)
  pcp followup_order_k       -> ORDER_REQUEST_VERIFIED(received[, named_owner]); received/unknown ->
                                COMMITMENT_BROKEN + COMMITMENT_MADE (same cid, new due: bumps followups)
  pcp request_addendum       -> ADDENDUM_REQUESTED(missing) + COMMITMENT_MADE
  pcp request_correction     -> CORRECTION_REQUESTED(fields) + COMMITMENT_MADE
  pcp send_order_<sid>       -> ORDER_SENT(sid) + COMMITMENT_MADE(fax to supplier)
  supplier screen            -> SUPPLIER_QUALIFIED | SUPPLIER_DISQUALIFIED (+ASSIGNMENT_REFUSED) |
                                SUPPLIER_NO_ANSWER (no answer / first unknown answer -> one re-screen)
  supplier commit            -> ASSIGNMENT_CONFIRMED + SUPPLIER_COMMITTED(ref, now+72h) | DISQUALIFIED
  supplier verify_order_received -> ORDER_RECEIVED
  supplier reverify          -> SUPPLIER_REVERIFIED | SUPPLIER_RELEASED (+ORDER_REDIRECT)
  supplier propose_slot      -> DELIVERY_SLOT_PROPOSED(slot)
  supplier verify_delivered  -> DELIVERED | DELIVERY_MISSED
  docs: VISIT_NOTE/ADDENDUM  -> VISIT_NOTE_RECEIVED + LCD_EVALUATED (cumulative over notes)
        SIGNED_SWO           -> SWO_SIGNED_RECEIVED + SWO_EVALUATED
        DELIVERY_POD         -> DELIVERED
An empty domain-event list from interpret_call means "nothing learned" (executor retries the task).
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any

from miramace import rules
from miramace.clock import add_business_hours
from miramace.events import Event, EventType, Source, make_event
from miramace.extract.extractor import normalize, quote_ok
from miramace.models import (
    CHICAGO,
    PCP_PARTY_ID,
    CallResult,
    CaseState,
    Claim,
    DeliveryStatus,
    InboundDoc,
    InboundDocKind,
    LCDChecklist,
    OrderStatus,
    SupplierFacts,
    SupplierStatus,
    Task,
    VisitNote,
)
from miramace.reducer import best_qualified

ET = EventType
LEASE_H = 72
DEFAULT_PROMISE_BH = 24

# ------------------------------------------------------------------------------------------ parsing

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_NUMWORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()
    )
}
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])")
_IN_DAYS_RE = re.compile(r"\b(?:in|within)?\s*(\d+|[a-z]+)\s+(business\s+|working\s+)?days?\b")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b")


def parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    t = text.lower()
    m = re.search(r"-?\d+", t)
    if m:
        return int(m.group(0))
    for w, n in _NUMWORDS.items():
        if re.search(rf"\b{w}\b", t):
            return n
    return None


def _add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def parse_when(text: str | None, now: datetime, *, default_hour: int = 17) -> datetime | None:
    """Relative/absolute day (+ optional time) -> tz-aware datetime, resolved against the virtual clock.

    Understands: today, tomorrow, end of day, weekday names (next occurrence after today, "next X" too),
    "in N (business) days", ISO dates, "Nov 5", times like "10am"/"2:30 pm"/"noon". Date without a time uses
    `default_hour` (17 = "by" semantics for promises). Time without a date: today if still ahead, else
    tomorrow. Returns None when nothing date-like is found.
    """
    if not text:
        return None
    t = text.lower().replace("’", "'")
    now = now.astimezone(CHICAGO)
    today = now.date()

    hour: int | None = None
    minute = 0
    m = _TIME_RE.search(t)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3).startswith("p"):
            hour += 12
    elif "noon" in t:
        hour = 12
    elif re.search(r"\b(end of (the )?(business )?day|eod|close of business|cob)\b", t):
        hour = 17

    day: date | None = None
    if (iso := _ISO_RE.search(t)) is not None:
        try:
            day = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            day = None
    elif (md := _MONTH_DAY_RE.search(t)) is not None:
        mon = _MONTHS.index(md.group(1)[:3]) + 1
        try:
            day = date(today.year, mon, int(md.group(2)))
            if day < today:
                day = date(today.year + 1, mon, int(md.group(2)))
        except ValueError:
            day = None
    elif re.search(r"\btomorrow\b", t):
        day = today + timedelta(days=1)
    elif re.search(r"\b(today|tonight|this afternoon|this morning)\b", t):
        day = today
    else:
        wd = next((i for i, w in enumerate(_WEEKDAYS) if re.search(rf"\b{w[:3]}(?:{w[3:]})?\b", t)), None)
        if wd is not None:
            delta = (wd - today.weekday()) % 7 or 7
            day = today + timedelta(days=delta)
        else:
            dm = _IN_DAYS_RE.search(t)
            if dm:
                n = parse_int(dm.group(1))
                if n is not None:
                    day = _add_business_days(today, n) if dm.group(2) else today + timedelta(days=n)
    if day is None and hour is None:
        return None
    if day is None:
        cand = datetime.combine(today, time(hour or 0, minute), tzinfo=CHICAGO)
        return cand if cand > now else cand + timedelta(days=1)
    return datetime.combine(day, time(default_hour if hour is None else hour, minute), tzinfo=CHICAGO)


# ------------------------------------------------------------------------------------------ helpers


class _Out:
    """Collects events with one shared source/time."""

    def __init__(self, case_id: str, source: Source, now: datetime) -> None:
        self.case_id, self.source, self.now = case_id, source, now
        self.events: list[Event] = []

    def add(self, type_: EventType, payload: Any = None) -> None:
        self.events.append(make_event(self.case_id, type_, payload, self.source, self.now))

    def failed(self, ref: str, field: str, reason: str) -> None:
        self.add(ET.EXTRACTION_FAILED, {"task_id": ref, "field": field, "reason": reason})


def verified_call_claims(result: CallResult, out: _Out, ref: str) -> dict[str, Claim]:
    """field -> stated claim whose quote is in ONE counterparty utterance (re-check; defence in depth)."""
    ok: dict[str, Claim] = {}
    for c in result.claims:
        if c.status != "stated":
            continue
        if not quote_ok(c.evidence_quote, result.transcript):
            out.failed(ref, c.field, "evidence quote not found in counterparty speech")
            continue
        ok.setdefault(c.field, c)
    return ok


def verified_doc_claims(doc: InboundDoc, claims: list[Claim], out: _Out) -> list[Claim]:
    nt = normalize(doc.text)
    good: list[Claim] = []
    for c in claims:
        if c.status != "stated":
            good.append(c)
            continue
        q = normalize(c.evidence_quote)
        if not q or q not in nt:
            out.failed(doc.doc_id, c.field, "evidence quote not found in document text")
            good.append(Claim(field=c.field, status="unknown", value=None, evidence_quote=None))
            continue
        good.append(c)
    return good


def _bool(claims: dict[str, Claim], field: str) -> bool | None:
    c = claims.get(field)
    return rules.parse_bool(c.value) if c else None


def _val(claims: dict[str, Claim], field: str) -> str | None:
    c = claims.get(field)
    return c.value.strip() if c and c.value else None


def _quote(claims: dict[str, Claim], *fields: str) -> str | None:
    for f in fields:
        c = claims.get(f)
        if c and c.evidence_quote:
            return c.evidence_quote
    return None


def _promise_due(claims: dict[str, Claim], now: datetime) -> datetime:
    due = parse_when(_val(claims, "promised_by"), now)
    if due is None or due <= now:
        due = add_business_hours(now, DEFAULT_PROMISE_BH)
    return due


def _commitment(out: _Out, cid: str, party: str, what: str, due: datetime, evidence: str | None) -> None:
    out.add(
        ET.COMMITMENT_MADE, {"cid": cid, "party": party, "what": what, "due_at": due, "evidence": evidence}
    )


def _release(out: _Out, state: CaseState, sid: str, reason: str) -> None:
    rec = state.suppliers.get(sid)
    if rec is None or rec.status in (SupplierStatus.RELEASED, SupplierStatus.DISQUALIFIED):
        return
    out.add(ET.SUPPLIER_RELEASED, {"sid": sid, "reason": reason})
    order = OrderStatus(state.order.state)
    if state.order_sent_to == sid and order in (OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY):
        backup = best_qualified(state, exclude=sid)
        if backup:
            out.add(ET.ORDER_REDIRECT, {"from_sid": sid, "to_sid": backup})


def release_events(state: CaseState, sid: str, reason: str, source: Source, now: datetime) -> list[Event]:
    """SUPPLIER_RELEASED (+ORDER_REDIRECT to the best backup) for an unreachable supplier."""
    out = _Out(state.case_id, source, now)
    _release(out, state, sid, reason)
    return out.events


def supplier_facts(claims: dict[str, Claim]) -> SupplierFacts:
    return SupplierFacts(
        taking_new_medicare=_bool(claims, "taking_new_medicare"),
        k0001_in_stock=_bool(claims, "k0001_in_stock"),
        delivers_to_zip=_bool(claims, "delivers_to_zip"),
        accepts_assignment_all_months=_bool(claims, "accepts_assignment_all_months"),
        earliest_delivery_days=parse_int(_val(claims, "earliest_delivery_days")),
        contact_name=_val(claims, "contact_name"),
        evidence={f: c.evidence_quote for f, c in claims.items() if c.evidence_quote},
    )


# ------------------------------------------------------------------------------------------ calls

_ANSWERED = ("COMPLETED",)


def interpret_call(
    task: Task, result: CallResult, state: CaseState, now: datetime, *, transcript_id: str | None = None
) -> list[Event]:
    """Domain events for one finished call. Audit events (EXTRACTION_FAILED) may be included."""
    ref = transcript_id or task.task_id
    out = _Out(state.case_id, Source(kind="TRANSCRIPT", ref=ref), now)
    claims = verified_call_claims(result, out, task.task_id)
    answered = result.outcome in _ANSWERED
    if task.party_id == PCP_PARTY_ID:
        _pcp(task, result, state, now, claims, answered, out)
    else:
        _supplier(task, result, state, now, claims, answered, out)
    return out.events


def _pcp(
    task: Task,
    result: CallResult,
    s: CaseState,
    now: datetime,
    cl: dict[str, Claim],
    answered: bool,
    out: _Out,
) -> None:
    intent = task.intent
    if result.outcome == "REFUSED" and intent == "request_order":
        from miramace.escalation import open_escalation
        from miramace.models import EscalationTrigger

        if not any(
            e.trigger == EscalationTrigger.PCP_REFUSED and e.status == "OPEN" for e in s.escalations.values()
        ):
            last = next((u.text for u in reversed(result.transcript) if u.speaker == "counterparty"), "")
            out.events.append(
                open_escalation(
                    EscalationTrigger.PCP_REFUSED,
                    s,
                    now,
                    {"reason": last or "declined", "key": task.idem_key},
                    source=out.source,
                )
            )
        return
    if intent.startswith("followup_order_"):
        _pcp_followup(task, s, now, cl, answered, out)
        return
    if not answered:
        return
    if intent == "request_order":
        out.add(ET.ORDER_REQUESTED, {"request_ref": task.idem_key, "version": task.version})
        what = _val(cl, "promised_action") or "fax the visit note and signed order"
        n = 1 + sum(1 for c in s.commitments if c.startswith(f"pcp-req-v{task.version}-"))
        _commitment(
            out,
            f"pcp-req-v{task.version}-{n}",
            PCP_PARTY_ID,
            what,
            _promise_due(cl, now),
            _quote(cl, "promised_by", "promised_action"),
        )
    elif intent == "request_addendum":
        if _bool(cl, "will_send_addendum") is False:
            return
        out.add(ET.ADDENDUM_REQUESTED, {"missing": task.args.get("missing", [])})
        _commitment(
            out,
            f"pcp:request_addendum:v{task.version}",
            PCP_PARTY_ID,
            "fax an addendum to the visit note",
            _promise_due(cl, now),
            _quote(cl, "promised_by", "will_send_addendum"),
        )
    elif intent == "request_correction":
        if _bool(cl, "will_send_correction") is False:
            return
        out.add(ET.CORRECTION_REQUESTED, {"fields": task.args.get("fields", [])})
        _commitment(
            out,
            f"pcp:request_correction:v{task.version}",
            PCP_PARTY_ID,
            "fax a corrected signed order",
            _promise_due(cl, now),
            _quote(cl, "promised_by", "will_send_correction"),
        )
    elif intent.startswith("send_order_"):
        sid = task.args.get("to_sid") or intent.removeprefix("send_order_")
        if _bool(cl, "will_fax_order") is not True:
            return
        if OrderStatus(s.order.state) != OrderStatus.VALID or s.committed_sid != sid:
            out.failed(task.task_id, "will_fax_order", f"order not sendable to {sid} (state changed)")
            return
        out.add(ET.ORDER_SENT, {"to_sid": sid})
        due = parse_when(_val(cl, "promised_by"), now)
        if due is None or due <= now:
            due = add_business_hours(now, 4)
        _commitment(
            out,
            f"pcp:send_order:{sid}:v{task.version}",
            PCP_PARTY_ID,
            f"fax the signed order to {sid}",
            due,
            _quote(cl, "will_fax_order"),
        )


def _pcp_followup(
    task: Task, s: CaseState, now: datetime, cl: dict[str, Claim], answered: bool, out: _Out
) -> None:
    """Verify first. Not received -> ORDER_REQUEST_VERIFIED(False) (planner resends a fresh version).
    Received with a new promised date -> COMMITMENT_BROKEN + COMMITMENT_MADE (same cid: followups advance,
    re-opened with the new due). No answer / no new promise -> COMMITMENT_BROKEN only (F1 rule b)."""
    cid = task.args.get("cid")
    open_ = cid in s.commitments and s.commitments[cid].status == "OPEN"
    # F1 contract: an unanswered follow-up advances the ladder even when the promise is already BROKEN.
    live = cid in s.commitments and s.commitments[cid].status in ("OPEN", "BROKEN")
    if not answered:
        if live:
            out.add(ET.COMMITMENT_BROKEN, {"cid": cid})
        return
    received = _bool(cl, "request_received")
    owner = _val(cl, "named_owner")
    if received is False:
        if OrderStatus(s.order.state) == OrderStatus.REQUESTED:
            out.add(ET.ORDER_REQUEST_VERIFIED, {"received": False, "named_owner": owner})
        elif live:
            out.add(ET.COMMITMENT_BROKEN, {"cid": cid})
        return
    if received is True:
        out.add(ET.ORDER_REQUEST_VERIFIED, {"received": True, "named_owner": owner})
    if not live:
        return
    out.add(ET.COMMITMENT_BROKEN, {"cid": cid})
    due = parse_when(_val(cl, "promised_by"), now)
    if due is not None and due > now:
        c = s.commitments[cid]
        what = c.what + (f" (owner: {owner})" if owner and "owner:" not in c.what else "")
        _commitment(out, cid, PCP_PARTY_ID, what, due, _quote(cl, "promised_by", "request_received"))


def _supplier(
    task: Task,
    result: CallResult,
    s: CaseState,
    now: datetime,
    cl: dict[str, Claim],
    answered: bool,
    out: _Out,
) -> None:
    sid, intent = task.party_id, task.intent
    rec = s.suppliers.get(sid)
    if intent == "screen":
        if rec is None or rec.status not in (
            SupplierStatus.UNCONTACTED,
            SupplierStatus.CALLING,
            SupplierStatus.NO_ANSWER,
        ):
            return
        if result.outcome == "REFUSED":
            out.add(ET.SUPPLIER_DISQUALIFIED, {"sid": sid, "reason": "refused to screen"})
            return
        if not answered:
            out.add(ET.SUPPLIER_NO_ANSWER, {"sid": sid, "attempt": rec.attempts + 1})
            return
        facts = supplier_facts(cl)
        ok, reason = rules.supplier_eligible(facts)
        if ok:
            out.add(ET.SUPPLIER_QUALIFIED, {"sid": sid, "facts": facts, "verified_at": now})
            return
        assert reason is not None
        if reason.startswith("unknown:") and rec.attempts == 0:
            # one re-screen: recorded as an unanswered attempt so the planner retries in 2bh
            out.failed(task.task_id, reason.split(":", 1)[1], "unknown after screening; one re-screen")
            out.add(ET.SUPPLIER_NO_ANSWER, {"sid": sid, "attempt": rec.attempts + 1})
            return
        if facts.accepts_assignment_all_months is False:
            out.add(ET.ASSIGNMENT_REFUSED, {"sid": sid})
        out.add(ET.SUPPLIER_DISQUALIFIED, {"sid": sid, "reason": reason})
        return
    if rec is None:
        return
    if not answered:
        if intent in ("verify_order_received", "reverify") and rec.status == SupplierStatus.COMMITTED:
            out.add(ET.SUPPLIER_NO_ANSWER, {"sid": sid, "attempt": task.attempts + 1})  # F1 rule c
        return
    if intent == "commit":
        if rec.status != SupplierStatus.QUALIFIED or s.committed_sid is not None:
            return
        reserve, assign = _bool(cl, "reserve_unit"), _bool(cl, "accepts_assignment_all_months")
        if reserve is False or assign is False:
            if assign is False:
                out.add(ET.ASSIGNMENT_REFUSED, {"sid": sid})
            out.add(
                ET.SUPPLIER_DISQUALIFIED,
                {
                    "sid": sid,
                    "reason": "not:accepts_assignment_all_months"
                    if assign is False
                    else "declined to reserve a unit",
                },
            )
            return
        if reserve is True and assign is True:
            ref = _val(cl, "reference_number") or f"{sid}-hold-v{task.version}"
            out.add(ET.ASSIGNMENT_CONFIRMED, {"sid": sid})
            out.add(
                ET.SUPPLIER_COMMITTED, {"sid": sid, "ref": ref, "expires_at": now + timedelta(hours=LEASE_H)}
            )
    elif intent == "verify_order_received":
        if (
            _bool(cl, "order_received") is True
            and OrderStatus(s.order.state) == OrderStatus.SENT_TO
            and s.order_sent_to == sid
        ):
            out.add(ET.ORDER_RECEIVED, {"by_sid": sid})
    elif intent == "reverify":
        stock, assign = _bool(cl, "k0001_in_stock"), _bool(cl, "accepts_assignment_all_months")
        if stock is False or assign is False:
            if assign is False:
                out.add(ET.ASSIGNMENT_REFUSED, {"sid": sid})
            _release(
                out,
                s,
                sid,
                "re-verification failed: " + ("out of stock" if stock is False else "no assignment"),
            )
        elif stock is True and assign is True:
            out.add(ET.SUPPLIER_REVERIFIED, {"sid": sid, "verified_at": now})
    elif intent == "propose_slot":
        slot = parse_when(_val(cl, "delivery_slot"), now, default_hour=10)
        if slot is None or slot <= now:
            if "delivery_slot" in cl:
                out.failed(
                    task.task_id,
                    "delivery_slot",
                    f"could not parse a future time from {_val(cl, 'delivery_slot')!r}",
                )
            return
        if s.committed_sid == sid and DeliveryStatus(s.delivery.state) in (
            DeliveryStatus.NOT_STARTED,
            DeliveryStatus.SCHEDULING,
        ):
            out.add(ET.DELIVERY_SLOT_PROPOSED, {"sid": sid, "slot": slot})
    elif intent == "verify_delivered":
        d = _bool(cl, "delivered")
        if DeliveryStatus(s.delivery.state) != DeliveryStatus.SCHEDULED or s.committed_sid != sid:
            return
        if d is True:
            out.add(ET.DELIVERED, {"sid": sid, "at": now})
        elif d is False:
            out.add(ET.DELIVERY_MISSED, None)


# ------------------------------------------------------------------------------------------ docs


def _prior_lcd_claims(lcd: LCDChecklist | None) -> list[Claim]:
    if lcd is None:
        return []
    return [
        Claim(field=c.value, status="stated", value=res, evidence_quote=lcd.evidence.get(c))
        for c, res in lcd.results.items()
        if res != "not_documented" and lcd.evidence.get(c)
    ]


def interpret_doc(doc: InboundDoc, claims: list[Claim], state: CaseState, now: datetime) -> list[Event]:
    out = _Out(state.case_id, Source(kind="INBOUND_DOC", ref=doc.doc_id), now)
    good = verified_doc_claims(doc, claims, out)
    if doc.kind in (InboundDocKind.VISIT_NOTE, InboundDocKind.ADDENDUM):
        note = VisitNote(
            doc_id=doc.doc_id, text=doc.text, received_at=now, is_addendum=doc.kind == InboundDocKind.ADDENDUM
        )
        out.add(ET.VISIT_NOTE_RECEIVED, {"note": note})
        checklist = rules.lcd_evaluate(_prior_lcd_claims(state.lcd) + good)
        out.add(ET.LCD_EVALUATED, {"checklist": checklist, "missing": checklist.missing})
    elif doc.kind == InboundDocKind.SIGNED_SWO:
        from miramace.fakes.sim import swo_from_claims

        swo = swo_from_claims(good)
        out.add(ET.SWO_SIGNED_RECEIVED, {"swo": swo})
        missing = rules.swo_completeness(swo)
        out.add(ET.SWO_EVALUATED, {"complete": not missing, "missing": missing, "hcpcs": swo.hcpcs})
    elif doc.kind == InboundDocKind.DELIVERY_POD:
        sid = doc.meta.get("sid")
        if (
            sid
            and state.committed_sid == sid
            and DeliveryStatus(state.delivery.state) == DeliveryStatus.SCHEDULED
        ):
            out.add(ET.DELIVERED, {"sid": sid, "at": now})
    return out.events


AUDIT = {ET.EXTRACTION_FAILED, ET.CALL_COMPLETED}


def learned_something(events: list[Event]) -> bool:
    """True if the events advance the case (not only audit / no-answer bookkeeping)."""
    return any(
        e.type not in AUDIT and e.type not in (ET.SUPPLIER_NO_ANSWER, ET.COMMITMENT_BROKEN) for e in events
    )
