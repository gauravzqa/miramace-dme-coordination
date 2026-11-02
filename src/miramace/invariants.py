"""B1 State — architecture invariants as assert functions; check_all(...).

Also hosts `schedule_blockers`, the invariant-1 predicate shared with reducer's DELIVERY_SCHEDULED guard.
Imports only frozen modules (models, events, clock) so reducer may import it without a cycle.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from miramace.clock import is_business, is_patient_time
from miramace.events import Event, EventType, OrderRequested, OrderRequestVerified, PatientNotified
from miramace.models import (
    TARGET_HCPCS,
    AssignmentStatus,
    CaseState,
    CaseStatus,
    DeliveryStatus,
    EscalationStatus,
    OrderStatus,
    PatientProfile,
    PatientStatus,
    SupplierStatus,
    Task,
    TaskStatus,
)

ET = EventType

VERIFY_WINDOW = timedelta(hours=24)
CALL_GRACE = timedelta(minutes=30)  # CALL_COMPLETED is stamped at call end; allow a call to run past 17:00
TERMINAL_CASE = {CaseStatus.DONE, CaseStatus.CANCELLED, CaseStatus.COVERAGE_CHANGED}
ORDER_VALID_OR_LATER = {OrderStatus.VALID, OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY}
SCHEDULED_OR_LATER = {DeliveryStatus.SCHEDULED, DeliveryStatus.DELIVERED, DeliveryStatus.POD_CONFIRMED}


class InvariantViolation(AssertionError):
    """An architecture invariant does not hold."""


def _fail(name: str, msg: str) -> None:
    raise InvariantViolation(f"[{name}] {msg}")


# ------------------------------------------------------------------------------------ invariant-1 predicate


def schedule_blockers(
    state: CaseState, sid: str | None, at: datetime | None, *, strict_patient: bool = True
) -> list[str]:
    """Reasons delivery cannot be (or could not have been) scheduled with `sid` at `at`. [] = OK.

    strict_patient=True requires patient.state == COST_ACCEPTED (reducer guard); False accepts any patient
    state once cost was accepted (invariant check after delivery moves the patient on).
    `at=None` skips the verified_at freshness check.
    """
    r: list[str] = []
    c = state.committed_sid
    if c is None:
        return ["no committed supplier"]
    if sid != c:
        r.append(f"sid {sid!r} is not the committed supplier {c!r}")
    if state.order_received_by != c:
        r.append(f"order received by {state.order_received_by!r}, not {c!r}")
    if state.order.state not in ORDER_VALID_OR_LATER:
        r.append(f"order is {state.order.state}, not VALID or later")
    if state.swo is None or not state.swo.signed or state.swo.hcpcs != TARGET_HCPCS:
        r.append(f"signed SWO with {TARGET_HCPCS} missing")
    if state.lcd is None or not state.lcd.all_met:
        r.append("LCD criteria not all met")
    if state.assignment_by_sid.get(c) != AssignmentStatus.CONFIRMED_ALL_MONTHS:
        r.append(f"assignment not confirmed for {c!r}")
    rec = state.suppliers.get(c)
    if at is not None:
        if rec is None or rec.verified_at is None:
            r.append("committed supplier never verified")
        elif at - rec.verified_at > VERIFY_WINDOW:
            r.append(f"verification stale ({rec.verified_at.isoformat()} > 24h before {at.isoformat()})")
    if strict_patient:
        if state.patient.state != PatientStatus.COST_ACCEPTED:
            r.append(f"patient is {state.patient.state}, not COST_ACCEPTED")
    elif "cost_accepted" not in state.patient_flags:
        r.append("patient never accepted cost")
    return r


# ------------------------------------------------------------------------------------ invariants


def assert_single_commit(state: CaseState) -> None:
    committed = [r.sid for r in state.suppliers.values() if r.status == SupplierStatus.COMMITTED]
    if len(committed) > 1:
        _fail("single_commit", f"{len(committed)} suppliers COMMITTED: {committed}")
    if committed and state.committed_sid != committed[0]:
        _fail("single_commit", f"committed_sid {state.committed_sid!r} != record {committed[0]!r}")


def assert_schedule_preconditions(state: CaseState, events: list[Event] | None = None) -> None:
    if state.delivery.state not in SCHEDULED_OR_LATER:
        return
    at = None
    if events:
        sched = [e for e in events if e.type == ET.DELIVERY_SCHEDULED]
        at = sched[-1].sim_time if sched else None
    blockers = schedule_blockers(state, state.committed_sid, at, strict_patient=False)
    if blockers:
        _fail("schedule_preconditions", f"delivery {state.delivery.state} but: " + "; ".join(blockers))


def assert_liveness(state: CaseState, tasks: list[Task], pending_inbound: bool) -> None:
    pending = [t for t in tasks if t.status == TaskStatus.PENDING]
    cs = state.case.state
    if cs in TERMINAL_CASE:
        if pending:
            _fail(
                "liveness", f"case {cs} has {len(pending)} PENDING task(s): {[t.idem_key for t in pending]}"
            )
        return
    if cs == CaseStatus.PAUSED:
        return  # waits on an explicit CASE_RESUMED
    open_esc = any(x.status == EscalationStatus.OPEN for x in state.escalations.values())
    if not (pending or open_esc or pending_inbound):
        _fail("liveness", f"case {cs} has no PENDING task, open escalation or pending inbound arrival")


def assert_idempotent_outbound(tasks: list[Task], events: list[Event]) -> None:
    dup = [k for k, n in Counter(t.idem_key for t in tasks).items() if n > 1]
    if dup:
        _fail("idempotent_outbound", f"duplicate idem keys: {dup}")
    # resend (request_order v>1) only after a verify step said "not received" (or a human decision)
    resend_ok_for: set[int] = set()  # versions v such that v+1 is a legal resend
    last_version = 0
    for e in events:
        if e.type == ET.ORDER_REQUESTED:
            p = e.parsed()
            assert isinstance(p, OrderRequested)
            if p.version > 1 and (p.version - 1) not in resend_ok_for:
                _fail(
                    "idempotent_outbound", f"ORDER_REQUESTED v{p.version} without a prior not-received verify"
                )
            last_version = p.version
        elif e.type == ET.ORDER_REQUEST_VERIFIED:
            p = e.parsed()
            assert isinstance(p, OrderRequestVerified)
            if not p.received and last_version:
                resend_ok_for.add(last_version)
        elif e.type == ET.HUMAN_RESOLVED and last_version:
            resend_ok_for.add(last_version)
    for t in tasks:
        if t.intent == "request_order" and t.version > 1 and t.status != TaskStatus.CANCELLED:
            if (t.version - 1) not in resend_ok_for:
                _fail("idempotent_outbound", f"task {t.idem_key} resends without a not-received verify")


def assert_traceable(events: list[Event], transcript_ids: set[str]) -> None:
    for e in events:
        if not e.source.kind or not e.source.ref:
            _fail("traceable", f"event {e.seq} {e.type} has no source kind/ref")
        if e.source.kind == "TRANSCRIPT" and e.source.ref not in transcript_ids:
            _fail("traceable", f"event {e.seq} {e.type} cites unknown transcript {e.source.ref!r}")
        if e.type == ET.CALL_COMPLETED and e.payload.get("transcript_id") not in transcript_ids:
            _fail(
                "traceable",
                f"CALL_COMPLETED {e.seq} cites unknown transcript {e.payload.get('transcript_id')!r}",
            )


def assert_patient_contact_ok(events: list[Event], profile: PatientProfile | None) -> None:
    for e in events:
        if e.type == ET.PATIENT_NOTIFIED:
            p = e.parsed()
            assert isinstance(p, PatientNotified)
            if not p.grounding_ok:
                _fail("patient_contact", f"PATIENT_NOTIFIED {e.seq} not grounded")
            if not is_patient_time(e.sim_time, profile):
                _fail("patient_contact", f"PATIENT_NOTIFIED {e.seq} outside patient hours at {e.sim_time}")
        elif e.type == ET.CALL_COMPLETED:
            if not (is_business(e.sim_time) or is_business(e.sim_time - CALL_GRACE)):
                _fail("patient_contact", f"call {e.seq} outside business hours at {e.sim_time}")


def check_all(
    state: CaseState,
    tasks: list[Task],
    events: list[Event],
    *,
    now: datetime,
    transcript_ids: set[str],
    pending_inbound: bool,
    check_liveness: bool = True,
) -> None:
    """Run every invariant. `check_liveness=False` for the pre-plan check at loop start (no tasks yet)."""
    del now  # reserved: all time checks use event sim_time
    assert_single_commit(state)
    assert_schedule_preconditions(state, events)
    if check_liveness:
        assert_liveness(state, tasks, pending_inbound)
    assert_idempotent_outbound(tasks, events)
    assert_traceable(events, transcript_ids)
    profile = state.patient_info.profile if state.patient_info else None
    assert_patient_contact_ok(events, profile)
