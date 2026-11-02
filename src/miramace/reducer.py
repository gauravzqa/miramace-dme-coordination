"""B1 State — pure reducer: reduce(state, event) -> CaseState; reduce_all(events).

Rules
- Pure: never mutates its input (works on a deep copy) and never performs I/O.
- Guard failure raises IllegalTransition; store.append then rejects the event (nothing written).
- Harmless out-of-order / irrelevant combos are no-ops (the event is still "recorded": last_seq advances).
- Owners are derived from (track, state) after every event, then overridden to ADVOCATE while an OPEN
  escalation targets that track ("who holds the ball"). due_at is set by transitions (patient, delivery)
  or derived (order <- open PCP commitment, supplier <- lease expiry).
- The reducer never generates follow-up events; escalation effects are appended by the run loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from miramace.clock import add_business_hours
from miramace.events import (
    AddendumRequested,
    AddressConfirmed,
    AssignmentConfirmed,
    AssignmentRefused,
    CandidatesRanked,
    CaseOpened,
    CommitmentMade,
    CommitmentRef,
    CoverageChanged,
    Delivered,
    DeliveryScheduled,
    DeliverySlotProposed,
    EscalationOpened,
    Event,
    EventType,
    HumanResolved,
    LcdEvaluated,
    OrderReceived,
    OrderRedirect,
    OrderRequested,
    OrderRequestVerified,
    OrderSent,
    PatientNotified,
    SupplierCommitted,
    SupplierDisqualified,
    SupplierNoAnswer,
    SupplierQualified,
    SupplierReleased,
    SupplierReverified,
    SwoDrafted,
    SwoEvaluated,
    SwoSignedReceived,
    VisitNoteReceived,
)
from miramace.invariants import schedule_blockers
from miramace.models import (
    PCP,
    PCP_PARTY_ID,
    TARGET_HCPCS,
    AssignmentStatus,
    CaseState,
    CaseStatus,
    Commitment,
    CommitmentStatus,
    DeliveryStatus,
    EscalationLane,
    EscalationStatus,
    OrderStatus,
    Owner,
    Patient,
    PatientStatus,
    SupplierRecord,
    SupplierStatus,
    SupplierTrackStatus,
    Track,
)

ET = EventType

MAX_SCREEN_ATTEMPTS = 3  # NO_ANSWER attempts before a supplier record becomes UNREACHABLE
DELIVERY_GRACE = timedelta(hours=4)  # SCHEDULED due = slot + 4h
COST_DECISION_BH = 24  # COST_DISCLOSED due = sim_time + 24 business hours

TERMINAL_CASE = {CaseStatus.DONE, CaseStatus.CANCELLED, CaseStatus.COVERAGE_CHANGED}

PATIENT_OWNER = {
    PatientStatus.INTAKE: Owner.PATIENT,
    PatientStatus.CONSENTED: Owner.US,
    PatientStatus.COST_DISCLOSED: Owner.PATIENT,
    PatientStatus.COST_ACCEPTED: Owner.US,
    PatientStatus.CONCERN_RAISED: Owner.ADVOCATE,
    PatientStatus.DELIVERY_CONFIRMED: Owner.NONE,
}
ORDER_OWNER = {
    OrderStatus.NOT_SENT: Owner.US,
    OrderStatus.REQUESTED: Owner.PCP,
    OrderStatus.DOCS_CHECK: Owner.US,
    OrderStatus.ADDENDUM_REQUESTED: Owner.PCP,
    OrderStatus.SIGNED: Owner.US,
    OrderStatus.CORRECTION_REQUESTED: Owner.PCP,
    OrderStatus.VALID: Owner.US,
    OrderStatus.SENT_TO: Owner.SUPPLIER,
    OrderStatus.RECEIVED_BY: Owner.US,
    OrderStatus.BLOCKED_ON_HUMAN: Owner.ADVOCATE,
}
DELIVERY_OWNER = {
    DeliveryStatus.NOT_STARTED: Owner.US,
    DeliveryStatus.SCHEDULING: Owner.PATIENT,
    DeliveryStatus.SCHEDULED: Owner.SUPPLIER,
    DeliveryStatus.DELIVERED: Owner.PATIENT,
    DeliveryStatus.POD_CONFIRMED: Owner.NONE,
}
CASE_OWNER = {
    CaseStatus.ACTIVE: Owner.US,
    CaseStatus.PAUSED: Owner.PATIENT,
    CaseStatus.CANCELLED: Owner.NONE,
    CaseStatus.COVERAGE_CHANGED: Owner.NONE,
    CaseStatus.DONE: Owner.NONE,
}
# order states where the PCP holds the ball (due = open PCP commitment due)
ORDER_PCP_DUE_STATES = {
    OrderStatus.REQUESTED,
    OrderStatus.ADDENDUM_REQUESTED,
    OrderStatus.CORRECTION_REQUESTED,
    OrderStatus.SENT_TO,
}
ORDER_REQUEST_FROM = {OrderStatus.NOT_SENT, OrderStatus.REQUESTED, OrderStatus.BLOCKED_ON_HUMAN}
ORDER_SIGNED_FROM = {
    OrderStatus.REQUESTED,
    OrderStatus.DOCS_CHECK,
    OrderStatus.ADDENDUM_REQUESTED,
    OrderStatus.CORRECTION_REQUESTED,
    OrderStatus.SIGNED,
}
PRE_QUALIFIED = {SupplierStatus.UNCONTACTED, SupplierStatus.CALLING, SupplierStatus.NO_ANSWER}


class IllegalTransition(Exception):
    """Guard failure: the event is not legal in the current state. store.append rejects it."""

    def __init__(self, event_type: str, reason: str) -> None:
        super().__init__(f"{event_type}: {reason}")
        self.event_type = event_type
        self.reason = reason


# ------------------------------------------------------------------------------------------ public API


def initial_state(case_id: str, patient: Patient, pcp: PCP) -> CaseState:
    s = CaseState(case_id=case_id, patient_info=patient.model_copy(deep=True), pcp=pcp.model_copy(deep=True))
    _finalize(s)
    return s


def reduce_all(events: list[Event]) -> CaseState:
    if not events:
        raise IllegalTransition("-", "empty event list")
    first = events[0]
    if first.type != ET.CASE_OPENED:
        raise IllegalTransition(first.type, "first event must be CASE_OPENED")
    p = first.parsed()
    assert isinstance(p, CaseOpened)
    state = initial_state(first.case_id, p.patient, p.pcp)
    for e in events:
        state = reduce(state, e)
    return state


def reduce(state: CaseState, event: Event) -> CaseState:
    """Return a NEW CaseState with `event` applied. Raises IllegalTransition on guard failure."""
    if event.case_id != state.case_id:
        raise IllegalTransition(event.type, f"case_id {event.case_id!r} != {state.case_id!r}")
    s = state.model_copy(deep=True)
    handler = _HANDLERS.get(event.type)
    if handler is not None:
        handler(s, event)
    s.last_seq = event.seq if event.seq else state.last_seq + 1
    _finalize(s)
    return s


# ------------------------------------------------------------------------------------------ helpers


def _illegal(e: Event, reason: str) -> IllegalTransition:
    return IllegalTransition(e.type, reason)


def _record(s: CaseState, e: Event, sid: str) -> SupplierRecord:
    rec = s.suppliers.get(sid)
    if rec is None:
        raise _illegal(e, f"unknown supplier {sid!r} (not ranked)")
    return rec


def _open_escalations(s: CaseState, track: Track | None = None) -> list:
    return [
        x
        for x in s.escalations.values()
        if x.status == EscalationStatus.OPEN and (track is None or x.track == track)
    ]


def _close_commitments(s: CaseState, party: str, status: CommitmentStatus = CommitmentStatus.KEPT) -> None:
    for c in s.commitments.values():
        if c.party == party and c.status == CommitmentStatus.OPEN:
            c.status = status


def _pcp_due(s: CaseState) -> datetime | None:
    dues = [
        c.due_at
        for c in s.commitments.values()
        if c.party == PCP_PARTY_ID and c.status == CommitmentStatus.OPEN
    ]
    return max(dues) if dues else None


def best_qualified(s: CaseState, exclude: str | None = None) -> str | None:
    """Best QUALIFIED supplier: assignment-confirmed first, then ranked order, then sid."""
    rank = {sid: i for i, sid in enumerate(s.ranked)}
    cands = [r.sid for r in s.suppliers.values() if r.status == SupplierStatus.QUALIFIED and r.sid != exclude]
    if not cands:
        return None
    cands.sort(
        key=lambda sid: (
            s.assignment_by_sid.get(sid) != AssignmentStatus.CONFIRMED_ALL_MONTHS,
            rank.get(sid, len(rank)),
            sid,
        )
    )
    return cands[0]


def _derived_order_state(s: CaseState) -> OrderStatus:
    """Order state to resume to after the last order-track escalation resolves (data-derived)."""
    if s.order_received_by:
        return OrderStatus.RECEIVED_BY
    if s.order_sent_to:
        return OrderStatus.SENT_TO
    if s.swo is not None and s.swo.signed:
        return OrderStatus.SIGNED
    if s.visit_notes:
        return OrderStatus.DOCS_CHECK
    if s.order_request_version > 0:
        return OrderStatus.REQUESTED
    return OrderStatus.NOT_SENT


def _block_order(s: CaseState) -> None:
    s.order.state = OrderStatus.BLOCKED_ON_HUMAN


def _order_is(s: CaseState, *states: OrderStatus) -> bool:
    return s.order.state in states


def _reset_delivery(s: CaseState) -> None:
    if s.delivery.state in (DeliveryStatus.SCHEDULING, DeliveryStatus.SCHEDULED):
        s.delivery.state = DeliveryStatus.NOT_STARTED
        s.delivery.due_at = None
        s.delivery_slot = None


def _finalize(s: CaseState) -> None:
    """Re-derive supplier/assignment tracks, owners, party ids and derived due times."""
    # supplier track
    committed = [r for r in s.suppliers.values() if r.status == SupplierStatus.COMMITTED]
    s.committed_sid = committed[0].sid if committed else None
    best = best_qualified(s)
    primary = s.committed_sid or best
    s.backup_sid = best_qualified(s, exclude=primary) if primary else None
    sup = s.supplier
    if s.committed_sid:
        sup.state, sup.party_id = SupplierTrackStatus.COMMITTED, s.committed_sid
        sup.owner, sup.due_at = Owner.SUPPLIER, committed[0].expires_at
    else:
        sup.owner, sup.due_at = Owner.US, None
        retryable = [
            sid
            for sid in s.ranked
            if sid in s.suppliers
            and (
                s.suppliers[sid].status in (SupplierStatus.UNCONTACTED, SupplierStatus.CALLING)
                or (
                    s.suppliers[sid].status == SupplierStatus.NO_ANSWER
                    and s.suppliers[sid].attempts < MAX_SCREEN_ATTEMPTS
                )
            )
        ]
        if best:
            sup.state, sup.party_id = SupplierTrackStatus.QUALIFIED, best
        elif retryable:
            sup.state, sup.party_id = SupplierTrackStatus.CALLING, retryable[0]
        elif not s.suppliers:
            sup.state, sup.party_id = SupplierTrackStatus.NOT_STARTED, None
        else:
            sup.state, sup.party_id = SupplierTrackStatus.EXHAUSTED, None

    # assignment track reflects committed, else best qualified
    a_sid = primary
    s.assignment.state = (
        s.assignment_by_sid.get(a_sid, AssignmentStatus.UNKNOWN) if a_sid else (AssignmentStatus.UNKNOWN)
    )
    s.assignment.party_id = a_sid
    s.assignment.owner = Owner.US

    # owners from state maps
    s.case.owner = CASE_OWNER[CaseStatus(s.case.state)]
    s.patient.owner = PATIENT_OWNER[PatientStatus(s.patient.state)]
    o = OrderStatus(s.order.state)
    s.order.owner = ORDER_OWNER[o]
    if o in (OrderStatus.REQUESTED, OrderStatus.ADDENDUM_REQUESTED, OrderStatus.CORRECTION_REQUESTED):
        s.order.party_id = PCP_PARTY_ID
    elif o in (OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY):
        s.order.party_id = s.order_received_by or s.order_sent_to
    else:
        s.order.party_id = None
    s.order.due_at = _pcp_due(s) if o in ORDER_PCP_DUE_STATES else None
    s.delivery.owner = DELIVERY_OWNER[DeliveryStatus(s.delivery.state)]
    s.delivery.party_id = s.committed_sid if s.delivery.state != DeliveryStatus.NOT_STARTED else None

    # escalation override: an OPEN escalation on a track gives the ball to the advocate
    if CaseStatus(s.case.state) not in TERMINAL_CASE:
        for esc in _open_escalations(s):
            s.track(esc.track).owner = Owner.ADVOCATE


# ------------------------------------------------------------------------------------------ case


def _case_opened(s: CaseState, e: Event) -> None:
    if s.last_seq != 0:
        raise _illegal(e, "case already opened")
    p = e.parsed()
    assert isinstance(p, CaseOpened)
    s.patient_info, s.pcp = p.patient, p.pcp


def _case_paused(s: CaseState, e: Event) -> None:
    if s.case.state != CaseStatus.ACTIVE:
        raise _illegal(e, f"case is {s.case.state}, not ACTIVE")
    s.case.state = CaseStatus.PAUSED  # lease release is emitted by the run loop as SUPPLIER_RELEASED


def _case_resumed(s: CaseState, e: Event) -> None:
    if s.case.state != CaseStatus.PAUSED:
        raise _illegal(e, f"case is {s.case.state}, not PAUSED")
    s.case.state = CaseStatus.ACTIVE


def _case_cancelled(s: CaseState, e: Event) -> None:
    if s.case.state == CaseStatus.DONE:
        raise _illegal(e, "case is DONE")
    s.case.state = CaseStatus.CANCELLED


def _coverage_changed(s: CaseState, e: Event) -> None:
    if s.case.state in (CaseStatus.DONE, CaseStatus.CANCELLED):
        raise _illegal(e, f"case is {s.case.state}")
    p = e.parsed()
    assert isinstance(p, CoverageChanged)
    if s.patient_info is not None:
        s.patient_info.coverage = p.new
    s.case.state = CaseStatus.COVERAGE_CHANGED


def _case_done(s: CaseState, e: Event) -> None:
    if s.case.state != CaseStatus.ACTIVE:
        raise _illegal(e, f"case is {s.case.state}, not ACTIVE")
    if s.delivery.state != DeliveryStatus.POD_CONFIRMED:
        raise _illegal(e, f"delivery is {s.delivery.state}, not POD_CONFIRMED")
    s.case.state = CaseStatus.DONE


# ------------------------------------------------------------------------------------------ patient


def _maybe_consented(s: CaseState) -> None:
    if {"address", "consent"} <= s.patient_flags:
        if s.patient.state == PatientStatus.INTAKE:
            s.patient.state = PatientStatus.CONSENTED
            s.patient.due_at = None
        elif s.patient.state == PatientStatus.CONCERN_RAISED and s.resume_to == PatientStatus.INTAKE:
            s.resume_to = PatientStatus.CONSENTED


def _patient_notified(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, PatientNotified)
    s.last_notified_at = e.sim_time
    s.notify_attempts = 0 if p.answered else s.notify_attempts + 1
    if p.purpose in s.pending_notify:
        s.pending_notify.remove(p.purpose)


def _address_confirmed(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, AddressConfirmed)
    if s.patient_info is not None:
        s.patient_info.address = p.address
        if p.home_access:
            s.patient_info.profile.home_access = p.home_access
    s.patient_flags.add("address")
    _maybe_consented(s)


def _consent_given(s: CaseState, e: Event) -> None:
    s.patient_flags.add("consent")
    _maybe_consented(s)


def _cost_disclosed(s: CaseState, e: Event) -> None:
    st = s.patient.state
    if st == PatientStatus.INTAKE or (
        st == PatientStatus.CONCERN_RAISED and s.resume_to == PatientStatus.INTAKE
    ):
        raise _illegal(e, "patient has not consented")
    s.patient_flags.add("cost_disclosed")
    if st in (PatientStatus.CONSENTED, PatientStatus.COST_DISCLOSED):
        s.patient.state = PatientStatus.COST_DISCLOSED
        s.patient.due_at = add_business_hours(e.sim_time, COST_DECISION_BH)
    elif st == PatientStatus.CONCERN_RAISED and s.resume_to == PatientStatus.CONSENTED:
        s.resume_to = PatientStatus.COST_DISCLOSED
    # COST_ACCEPTED / DELIVERY_CONFIRMED: re-disclosure is a no-op


def _cost_accepted(s: CaseState, e: Event) -> None:
    if "cost_disclosed" not in s.patient_flags:
        raise _illegal(e, "COST_ACCEPTED before COST_DISCLOSED")
    s.patient_flags.add("cost_accepted")
    if s.patient.state == PatientStatus.COST_DISCLOSED:
        s.patient.state = PatientStatus.COST_ACCEPTED
        s.patient.due_at = None
    elif s.patient.state == PatientStatus.CONCERN_RAISED and s.resume_to in (
        PatientStatus.CONSENTED,
        PatientStatus.COST_DISCLOSED,
    ):
        s.resume_to = PatientStatus.COST_ACCEPTED


def _concern_raised(s: CaseState, e: Event) -> None:
    if s.patient.state != PatientStatus.CONCERN_RAISED:
        s.resume_to = s.patient.state
        s.patient.state = PatientStatus.CONCERN_RAISED


def _patient_pod(s: CaseState) -> None:
    if s.patient.state == PatientStatus.COST_ACCEPTED:
        s.patient.state = PatientStatus.DELIVERY_CONFIRMED
        s.patient.due_at = None
    elif s.patient.state == PatientStatus.CONCERN_RAISED and s.resume_to == PatientStatus.COST_ACCEPTED:
        s.resume_to = PatientStatus.DELIVERY_CONFIRMED


# ------------------------------------------------------------------------------------------ order


def _swo_drafted(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SwoDrafted)
    if s.swo is None or not s.swo.signed:
        s.swo = p.swo


def _order_requested(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, OrderRequested)
    if s.order.state not in ORDER_REQUEST_FROM:
        raise _illegal(e, f"order is {s.order.state}")
    if p.version <= s.order_request_version:
        raise _illegal(e, f"version {p.version} <= current {s.order_request_version}")
    s.order_request_version = p.version
    if s.order.state != OrderStatus.BLOCKED_ON_HUMAN:
        s.order.state = OrderStatus.REQUESTED


def _order_request_verified(s: CaseState, e: Event) -> None:
    """'We never got it' → the open PCP promise is broken and the order goes back to NOT_SENT,
    so the planner re-sends under the next version (verify-first resend, invariant 3)."""
    p = e.parsed()
    assert isinstance(p, OrderRequestVerified)
    if p.received:
        return
    if s.order.state == OrderStatus.REQUESTED:
        _close_commitments(s, PCP_PARTY_ID, CommitmentStatus.BROKEN)
        s.order.state = OrderStatus.NOT_SENT


def _visit_note_received(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, VisitNoteReceived)
    s.visit_notes.append(p.note)
    _close_commitments(s, PCP_PARTY_ID)
    if _order_is(s, OrderStatus.REQUESTED, OrderStatus.ADDENDUM_REQUESTED):
        s.order.state = OrderStatus.DOCS_CHECK


def _lcd_evaluated(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, LcdEvaluated)
    s.lcd = p.checklist


def _addendum_requested(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, AddendumRequested)
    if s.order.state == OrderStatus.DOCS_CHECK:
        s.order.state = OrderStatus.ADDENDUM_REQUESTED
    elif s.order.state not in (
        OrderStatus.SIGNED,
        OrderStatus.CORRECTION_REQUESTED,
        OrderStatus.BLOCKED_ON_HUMAN,
    ):
        raise _illegal(e, f"order is {s.order.state}")
    s.addendum_requests += 1


def _swo_signed_received(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SwoSignedReceived)
    st = s.order.state
    if st not in ORDER_SIGNED_FROM and st != OrderStatus.BLOCKED_ON_HUMAN:
        raise _illegal(e, f"order is {st}")
    s.swo = p.swo
    s.swo_missing = []  # re-evaluated by SWO_EVALUATED
    _close_commitments(s, PCP_PARTY_ID)
    if st != OrderStatus.BLOCKED_ON_HUMAN:
        s.order.state = OrderStatus.SIGNED


def _swo_evaluated(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SwoEvaluated)
    s.swo_missing = list(p.missing)


def _correction_requested(s: CaseState, e: Event) -> None:
    if s.order.state == OrderStatus.SIGNED:
        s.order.state = OrderStatus.CORRECTION_REQUESTED
    elif s.order.state != OrderStatus.BLOCKED_ON_HUMAN:
        raise _illegal(e, f"order is {s.order.state}, not SIGNED")
    s.correction_requests += 1


def _order_validated(s: CaseState, e: Event) -> None:
    if s.order.state == OrderStatus.VALID:
        return
    if s.order.state != OrderStatus.SIGNED:
        raise _illegal(e, f"order is {s.order.state}, not SIGNED")
    reasons = []
    if s.lcd is None or not s.lcd.all_met:
        reasons.append("LCD not all met")
    if s.swo is None or not s.swo.signed:
        reasons.append("SWO not signed")
    elif s.swo.hcpcs != TARGET_HCPCS:
        reasons.append(f"hcpcs {s.swo.hcpcs} != {TARGET_HCPCS}")
    if s.swo_missing:
        reasons.append(f"SWO missing {s.swo_missing}")
    if reasons:
        raise _illegal(e, "; ".join(reasons))
    s.order.state = OrderStatus.VALID


def _order_sent(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, OrderSent)
    if s.order.state != OrderStatus.VALID:
        raise _illegal(e, f"order is {s.order.state}, not VALID")
    if p.to_sid != s.committed_sid:
        raise _illegal(e, f"{p.to_sid!r} is not the committed supplier ({s.committed_sid!r})")
    s.order_sent_to = p.to_sid
    s.order.state = OrderStatus.SENT_TO


def _order_received(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, OrderReceived)
    if s.order.state == OrderStatus.RECEIVED_BY and s.order_received_by == p.by_sid:
        return
    if s.order.state != OrderStatus.SENT_TO:
        raise _illegal(e, f"order is {s.order.state}, not SENT_TO")
    if p.by_sid != s.order_sent_to:
        raise _illegal(e, f"order was sent to {s.order_sent_to!r}, not {p.by_sid!r}")
    s.order_received_by = p.by_sid
    s.order.state = OrderStatus.RECEIVED_BY
    _close_commitments(s, PCP_PARTY_ID)
    _close_commitments(s, p.by_sid)


def _order_redirect(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, OrderRedirect)
    if s.order.state not in (OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY):
        raise _illegal(e, f"order is {s.order.state}, not SENT_TO/RECEIVED_BY")
    _close_commitments(s, p.from_sid, CommitmentStatus.CANCELLED)
    s.order_sent_to = None
    s.order_received_by = None
    s.order.state = OrderStatus.VALID
    _reset_delivery(s)


def _order_blocked(s: CaseState, e: Event) -> None:
    _block_order(s)


# ------------------------------------------------------------------------------------------ supplier


def _candidates_ranked(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, CandidatesRanked)
    s.ranked = list(dict.fromkeys(p.sids))
    for sid in s.ranked:
        s.suppliers.setdefault(sid, SupplierRecord(sid=sid))


def _supplier_no_answer(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierNoAnswer)
    rec = _record(s, e, p.sid)
    if rec.status not in PRE_QUALIFIED:
        return  # e.g. no answer on a reverify call: tracked by the planner, not here
    rec.attempts = max(rec.attempts + 1, p.attempt)
    rec.status = (
        SupplierStatus.UNREACHABLE if rec.attempts >= MAX_SCREEN_ATTEMPTS else SupplierStatus.NO_ANSWER
    )


def _supplier_qualified(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierQualified)
    rec = _record(s, e, p.sid)
    if rec.status == SupplierStatus.RELEASED:
        return  # a released (ghosted) supplier is never re-qualified
    rec.facts, rec.verified_at, rec.reason = p.facts, p.verified_at, None
    if rec.status != SupplierStatus.COMMITTED:
        rec.status = SupplierStatus.QUALIFIED
    aa = p.facts.accepts_assignment_all_months
    if aa is True and s.assignment_by_sid.get(p.sid) != AssignmentStatus.REFUSED:
        s.assignment_by_sid[p.sid] = AssignmentStatus.CONFIRMED_ALL_MONTHS
    elif aa is False:
        s.assignment_by_sid[p.sid] = AssignmentStatus.REFUSED


def _supplier_disqualified(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierDisqualified)
    rec = _record(s, e, p.sid)
    if rec.status == SupplierStatus.COMMITTED:
        raise _illegal(e, "release a COMMITTED supplier before disqualifying it")
    rec.status, rec.reason = SupplierStatus.DISQUALIFIED, p.reason


def _supplier_committed(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierCommitted)
    rec = _record(s, e, p.sid)
    if s.committed_sid is not None:
        raise _illegal(e, f"{s.committed_sid!r} is already COMMITTED")
    if rec.status != SupplierStatus.QUALIFIED:
        raise _illegal(e, f"{p.sid!r} is {rec.status}, not QUALIFIED")
    if s.assignment_by_sid.get(p.sid) != AssignmentStatus.CONFIRMED_ALL_MONTHS:
        raise _illegal(e, f"assignment not confirmed for {p.sid!r}")
    rec.status, rec.ref, rec.expires_at = SupplierStatus.COMMITTED, p.ref, p.expires_at


def _supplier_reverified(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierReverified)
    rec = _record(s, e, p.sid)
    if rec.status in (SupplierStatus.QUALIFIED, SupplierStatus.COMMITTED):
        rec.verified_at = p.verified_at


def _supplier_released(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, SupplierReleased)
    rec = _record(s, e, p.sid)
    was_committed = rec.status == SupplierStatus.COMMITTED
    rec.status, rec.reason = SupplierStatus.RELEASED, p.reason
    _close_commitments(s, p.sid, CommitmentStatus.CANCELLED)
    if was_committed:
        _reset_delivery(s)


def _assignment_confirmed(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, AssignmentConfirmed)
    _record(s, e, p.sid)
    s.assignment_by_sid[p.sid] = AssignmentStatus.CONFIRMED_ALL_MONTHS


def _assignment_refused(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, AssignmentRefused)
    _record(s, e, p.sid)
    s.assignment_by_sid[p.sid] = AssignmentStatus.REFUSED


# ------------------------------------------------------------------------------------------ delivery


def _slot_proposed(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, DeliverySlotProposed)
    if s.delivery.state not in (
        DeliveryStatus.NOT_STARTED,
        DeliveryStatus.SCHEDULING,
        DeliveryStatus.SCHEDULED,
    ):
        raise _illegal(e, f"delivery is {s.delivery.state}")
    if p.sid != s.committed_sid:
        raise _illegal(e, f"{p.sid!r} is not the committed supplier")
    s.delivery.state = DeliveryStatus.SCHEDULING
    s.delivery.due_at = None
    s.delivery_slot = p.slot


def _delivery_scheduled(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, DeliveryScheduled)
    if s.delivery.state != DeliveryStatus.SCHEDULING:
        raise _illegal(e, f"delivery is {s.delivery.state}, not SCHEDULING")
    blockers = schedule_blockers(s, p.sid, e.sim_time, strict_patient=True)
    if blockers:
        raise _illegal(e, "cannot schedule: " + "; ".join(blockers))
    s.delivery.state = DeliveryStatus.SCHEDULED
    s.delivery_slot = p.slot
    s.delivery.due_at = p.slot + DELIVERY_GRACE


def _delivery_missed(s: CaseState, e: Event) -> None:
    if s.delivery.state != DeliveryStatus.SCHEDULED:
        raise _illegal(e, f"delivery is {s.delivery.state}, not SCHEDULED")
    s.delivery.state = DeliveryStatus.SCHEDULING
    s.delivery.due_at = None
    s.delivery_slot = None


def _delivered(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, Delivered)
    if s.delivery.state != DeliveryStatus.SCHEDULED:
        raise _illegal(e, f"delivery is {s.delivery.state}, not SCHEDULED")
    if p.sid != s.committed_sid:
        raise _illegal(e, f"{p.sid!r} is not the committed supplier")
    s.delivery.state = DeliveryStatus.DELIVERED
    s.delivery.due_at = None
    _close_commitments(s, p.sid)


def _pod_confirmed(s: CaseState, e: Event) -> None:
    if s.delivery.state == DeliveryStatus.POD_CONFIRMED:
        return
    if s.delivery.state != DeliveryStatus.DELIVERED:
        raise _illegal(e, f"delivery is {s.delivery.state}, not DELIVERED")
    s.delivery.state = DeliveryStatus.POD_CONFIRMED
    _patient_pod(s)


# ------------------------------------------------------------------------------------------ ledger


def _commitment_made(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, CommitmentMade)
    prev = s.commitments.get(p.cid)
    s.commitments[p.cid] = Commitment(
        cid=p.cid,
        party=p.party,
        what=p.what,
        due_at=p.due_at,
        status=CommitmentStatus.OPEN,
        source_event_id=e.event_id,
        followups=prev.followups if prev else 0,
    )


def _commitment_kept(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, CommitmentRef)
    c = s.commitments.get(p.cid)
    if c is None:
        raise _illegal(e, f"unknown commitment {p.cid!r}")
    if c.status in (CommitmentStatus.OPEN, CommitmentStatus.BROKEN):
        c.status = CommitmentStatus.KEPT


def _commitment_broken(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, CommitmentRef)
    c = s.commitments.get(p.cid)
    if c is None:
        raise _illegal(e, f"unknown commitment {p.cid!r}")
    if c.status == CommitmentStatus.OPEN:
        c.status = CommitmentStatus.BROKEN
        c.followups += 1


# ------------------------------------------------------------------------------------------ escalation


def _escalation_opened(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, EscalationOpened)
    esc = p.escalation.model_copy(update={"status": EscalationStatus.OPEN, "resolution": None})
    if esc.eid in s.escalations:
        raise _illegal(e, f"escalation {esc.eid!r} already exists")
    s.escalations[esc.eid] = esc
    if esc.track == Track.ORDER:
        _block_order(s)


def _human_resolved(s: CaseState, e: Event) -> None:
    p = e.parsed()
    assert isinstance(p, HumanResolved)
    esc = s.escalations.get(p.eid)
    if esc is None:
        raise _illegal(e, f"unknown escalation {p.eid!r}")
    if esc.status != EscalationStatus.OPEN:
        raise _illegal(e, f"escalation {p.eid!r} already resolved")
    if esc.options and p.option_id not in {o.option_id for o in esc.options}:
        raise _illegal(e, f"option {p.option_id!r} not offered by {p.eid!r}")
    esc.status, esc.resolution = EscalationStatus.RESOLVED, p.option_id

    still_open = _open_escalations(s, esc.track)
    if esc.track == Track.ORDER and not still_open and s.order.state == OrderStatus.BLOCKED_ON_HUMAN:
        s.order.state = _derived_order_state(s)
    patient_side = esc.lane == EscalationLane.PATIENT or esc.track == Track.PATIENT
    if patient_side and s.patient.state == PatientStatus.CONCERN_RAISED:
        others = [
            x for x in _open_escalations(s) if x.lane == EscalationLane.PATIENT or x.track == Track.PATIENT
        ]
        if not others:
            s.patient.state = s.resume_to or PatientStatus.INTAKE
            s.resume_to = None


def _noop(s: CaseState, e: Event) -> None:
    """Audit / informational events: recorded (last_seq advances) without state change."""


_HANDLERS = {
    ET.CASE_OPENED: _case_opened,
    ET.CASE_PAUSED: _case_paused,
    ET.CASE_RESUMED: _case_resumed,
    ET.CASE_CANCELLED: _case_cancelled,
    ET.COVERAGE_CHANGED: _coverage_changed,
    ET.CASE_DONE: _case_done,
    ET.PATIENT_NOTIFIED: _patient_notified,
    ET.ADDRESS_CONFIRMED: _address_confirmed,
    ET.CONSENT_GIVEN: _consent_given,
    ET.COST_DISCLOSED: _cost_disclosed,
    ET.COST_ACCEPTED: _cost_accepted,
    ET.CONCERN_RAISED: _concern_raised,
    ET.STATUS_ASKED: _noop,
    ET.PATIENT_UNREACHABLE: _noop,
    ET.SWO_DRAFTED: _swo_drafted,
    ET.ORDER_REQUESTED: _order_requested,
    ET.ORDER_REQUEST_VERIFIED: _order_request_verified,
    ET.VISIT_NOTE_RECEIVED: _visit_note_received,
    ET.LCD_EVALUATED: _lcd_evaluated,
    ET.ADDENDUM_REQUESTED: _addendum_requested,
    ET.SWO_SIGNED_RECEIVED: _swo_signed_received,
    ET.SWO_EVALUATED: _swo_evaluated,
    ET.CORRECTION_REQUESTED: _correction_requested,
    ET.ORDER_VALIDATED: _order_validated,
    ET.ORDER_SENT: _order_sent,
    ET.ORDER_RECEIVED: _order_received,
    ET.ORDER_REDIRECT: _order_redirect,
    ET.ORDER_STALLED: _order_blocked,
    ET.PCP_REFUSED: _order_blocked,
    ET.CANDIDATES_RANKED: _candidates_ranked,
    ET.SUPPLIER_NO_ANSWER: _supplier_no_answer,
    ET.SUPPLIER_QUALIFIED: _supplier_qualified,
    ET.SUPPLIER_DISQUALIFIED: _supplier_disqualified,
    ET.SUPPLIER_COMMITTED: _supplier_committed,
    ET.SUPPLIER_REVERIFIED: _supplier_reverified,
    ET.SUPPLIER_RELEASED: _supplier_released,
    ET.SUPPLIERS_EXHAUSTED: _noop,
    ET.ASSIGNMENT_CONFIRMED: _assignment_confirmed,
    ET.ASSIGNMENT_REFUSED: _assignment_refused,
    ET.DELIVERY_SLOT_PROPOSED: _slot_proposed,
    ET.DELIVERY_SCHEDULED: _delivery_scheduled,
    ET.DELIVERY_MISSED: _delivery_missed,
    ET.DELIVERED: _delivered,
    ET.POD_CONFIRMED: _pod_confirmed,
    ET.COMMITMENT_MADE: _commitment_made,
    ET.COMMITMENT_KEPT: _commitment_kept,
    ET.COMMITMENT_BROKEN: _commitment_broken,
    ET.CALL_COMPLETED: _noop,
    ET.EXTRACTION_FAILED: _noop,
    ET.ESCALATION_OPENED: _escalation_opened,
    ET.HUMAN_RESOLVED: _human_resolved,
}
assert set(_HANDLERS) == set(EventType), "every EventType needs a reducer handler"
