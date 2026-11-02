"""B1 — reducer transition tables (legal + illegal) per track; also hosts shared B1 event builders."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from miramace.clock import SIM_START, add_business_hours
from miramace.events import Event, EventType, Source, make_event
from miramace.models import (
    SWO,
    Address,
    AssignmentStatus,
    CaseState,
    CaseStatus,
    CommitmentStatus,
    CostInfo,
    DeliveryStatus,
    Escalation,
    EscalationLane,
    EscalationOption,
    EscalationStatus,
    EscalationTrigger,
    LCDChecklist,
    LCDCriterion,
    OrderStatus,
    Owner,
    PatientStatus,
    SupplierFacts,
    SupplierStatus,
    SupplierTrackStatus,
    Track,
    VisitNote,
    dr_chen,
    eleanor,
)
from miramace.reducer import IllegalTransition, initial_state, reduce, reduce_all

ET = EventType
CASE = "case-1"
A, B, C = "alpha-dme", "bravo-dme", "charlie-dme"
ADDR = Address(line1="1820 N Clark St", city="Chicago", state="IL", zip="60614")
GOOD_FACTS = SupplierFacts(
    taking_new_medicare=True,
    k0001_in_stock=True,
    delivers_to_zip=True,
    accepts_assignment_all_months=True,
    earliest_delivery_days=2,
)
ALL_MET = LCDChecklist(results={c: "met" for c in LCDCriterion}, evidence={c: "q" for c in LCDCriterion})
SIGNED_SWO = SWO(
    beneficiary_name="Eleanor Martinez",
    mbi="1EG4-TE5-MK72",
    order_date=date(2026, 11, 3),
    item_description="Standard manual wheelchair",
    hcpcs="K0001",
    quantity=1,
    practitioner_name="Dr. Sarah Chen",
    practitioner_npi="1234567893",
    signed=True,
    signature_date=date(2026, 11, 3),
)


class Log:
    """Builds events with make_event at a monotonically advancing sim time."""

    def __init__(self, case_id: str = CASE, t: datetime = SIM_START) -> None:
        self.case_id, self.t, self.events = case_id, t, []

    def ev(self, type_: ET, payload=None, *, kind="RULE", ref="test", dt_min: int = 5) -> Event:
        self.t += timedelta(minutes=dt_min)
        e = make_event(self.case_id, type_, payload, Source(kind=kind, ref=ref), self.t)
        self.events.append(e)
        return e

    def add(self, type_: ET, payload=None, **kw) -> Log:
        self.ev(type_, payload, **kw)
        return self

    def state(self) -> CaseState:
        return reduce_all(self.events)


def opened(case_id: str = CASE) -> Log:
    return Log(case_id).add(ET.CASE_OPENED, {"patient": eleanor(), "pcp": dr_chen()})


def esc(eid: str, track: Track, trigger: EscalationTrigger, lane=EscalationLane.CASE, opts=("a", "b")):
    return Escalation(
        eid=eid,
        lane=lane,
        trigger=trigger,
        track=track,
        question="?",
        options=[EscalationOption(option_id=o, label=o) for o in opts],
    )


def consented(log: Log | None = None) -> Log:
    log = log or opened()
    return log.add(ET.ADDRESS_CONFIRMED, {"address": ADDR}).add(ET.CONSENT_GIVEN, {"scope": "coordination"})


def order_valid(log: Log) -> Log:
    return (
        log.add(ET.ORDER_REQUESTED, {"request_ref": "r1", "version": 1})
        .add(
            ET.COMMITMENT_MADE,
            {
                "cid": "c-pcp-1",
                "party": "pcp",
                "what": "fax note+SWO",
                "due_at": SIM_START + timedelta(days=1),
            },
        )
        .add(ET.VISIT_NOTE_RECEIVED, {"note": VisitNote(doc_id="n1", text="note", received_at=log.t)})
        .add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
        .add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO})
        .add(ET.SWO_EVALUATED, {"complete": True, "missing": [], "hcpcs": "K0001"})
        .add(ET.ORDER_VALIDATED)
    )


def committed(log: Log, sid: str = A) -> Log:
    return (
        log.add(ET.CANDIDATES_RANKED, {"sids": [A, B, C], "distances": {A: 1.0, B: 2.0, C: 3.0}})
        .add(ET.SUPPLIER_QUALIFIED, {"sid": A, "facts": GOOD_FACTS, "verified_at": log.t})
        .add(ET.SUPPLIER_QUALIFIED, {"sid": B, "facts": GOOD_FACTS, "verified_at": log.t})
        .add(ET.SUPPLIER_COMMITTED, {"sid": sid, "ref": "R-1", "expires_at": log.t + timedelta(hours=72)})
    )


def ready_to_schedule() -> Log:
    log = committed(order_valid(consented()))
    return (
        log.add(ET.COST_DISCLOSED, {"cost": CostInfo()})
        .add(ET.COST_ACCEPTED)
        .add(ET.ORDER_SENT, {"to_sid": A})
        .add(ET.ORDER_RECEIVED, {"by_sid": A})
        .add(ET.SUPPLIER_REVERIFIED, {"sid": A, "verified_at": log.t + timedelta(minutes=30)})
        .add(ET.DELIVERY_SLOT_PROPOSED, {"sid": A, "slot": log.t + timedelta(days=2)})
    )


def happy_path() -> Log:
    log = ready_to_schedule()
    slot = log.t + timedelta(days=2)
    return (
        log.add(ET.DELIVERY_SCHEDULED, {"sid": A, "slot": slot})
        .add(ET.DELIVERED, {"sid": A, "at": slot})
        .add(ET.POD_CONFIRMED)
        .add(ET.CASE_DONE)
    )


def apply(log: Log, type_: ET, payload=None) -> CaseState:
    st = log.state()
    e = make_event(CASE, type_, payload, Source(kind="RULE", ref="t"), log.t + timedelta(minutes=1))
    return reduce(st, e)


def illegal(log: Log, type_: ET, payload=None) -> None:
    with pytest.raises(IllegalTransition):
        apply(log, type_, payload)


# ------------------------------------------------------------------------------ basics


def test_reduce_all_requires_case_opened_first():
    with pytest.raises(IllegalTransition):
        reduce_all([Log().ev(ET.CASE_DONE)])
    with pytest.raises(IllegalTransition):
        reduce_all([])


def test_initial_state_and_seq():
    st = opened().state()
    assert st.case.state == CaseStatus.ACTIVE and st.last_seq == 1
    assert st.patient.state == PatientStatus.INTAKE and st.patient.owner == Owner.PATIENT
    assert st.supplier.state == SupplierTrackStatus.NOT_STARTED
    assert initial_state(CASE, eleanor(), dr_chen()).last_seq == 0


def test_reduce_is_pure_and_checks_case_id():
    log = opened()
    st = log.state()
    snapshot = st.model_dump()
    e = make_event(CASE, ET.ADDRESS_CONFIRMED, {"address": ADDR}, Source(kind="RULE", ref="t"), log.t)
    new = reduce(st, e)
    assert st.model_dump() == snapshot and new is not st and new.patient_info.address == ADDR
    other = make_event("other", ET.CONSENT_GIVEN, {"scope": "x"}, Source(kind="RULE", ref="t"), log.t)
    with pytest.raises(IllegalTransition):
        reduce(st, other)
    illegal(log, ET.CASE_OPENED, {"patient": eleanor(), "pcp": dr_chen()})


def test_audit_events_are_noops():
    log = opened()
    before = log.state()
    for t, p in [
        (ET.STATUS_ASKED, {"utterance": "how's it going"}),
        (ET.CALL_COMPLETED, {"task_id": "t", "transcript_id": "tr", "outcome": "COMPLETED"}),
        (ET.EXTRACTION_FAILED, {"task_id": "t", "field": "f", "reason": "r"}),
        (ET.ORDER_REQUEST_VERIFIED, {"received": True}),
    ]:
        log.add(t, p)
    after = log.state()
    assert after.model_dump(exclude={"last_seq"}) == before.model_dump(exclude={"last_seq"})
    assert after.last_seq == 5


# ------------------------------------------------------------------------------ case track


@pytest.mark.parametrize(
    "steps,expected",
    [
        ([(ET.CASE_PAUSED, {"reason": "hospital"})], CaseStatus.PAUSED),
        ([(ET.CASE_PAUSED, {"reason": "h"}), (ET.CASE_RESUMED, None)], CaseStatus.ACTIVE),
        ([(ET.CASE_CANCELLED, {"reason": "x"})], CaseStatus.CANCELLED),
        ([(ET.CASE_PAUSED, {"reason": "h"}), (ET.CASE_CANCELLED, {"reason": "x"})], CaseStatus.CANCELLED),
        ([(ET.COVERAGE_CHANGED, {"new": "MA_PLAN"})], CaseStatus.COVERAGE_CHANGED),
    ],
)
def test_case_legal(steps, expected):
    log = opened()
    for t, p in steps:
        log.add(t, p)
    st = log.state()
    assert st.case.state == expected
    if expected == CaseStatus.COVERAGE_CHANGED:
        assert st.patient_info.coverage == "MA_PLAN"


@pytest.mark.parametrize(
    "pre,bad",
    [
        ([], (ET.CASE_RESUMED, None)),
        ([], (ET.CASE_DONE, None)),  # CASE_DONE before POD_CONFIRMED
        ([(ET.CASE_PAUSED, {"reason": "h"})], (ET.CASE_PAUSED, {"reason": "h"})),
        ([(ET.CASE_PAUSED, {"reason": "h"})], (ET.CASE_DONE, None)),
    ],
)
def test_case_illegal(pre, bad):
    log = opened()
    for t, p in pre:
        log.add(t, p)
    illegal(log, *bad)


def test_case_done_terminal():
    st = happy_path().state()
    assert st.case.state == CaseStatus.DONE and st.case.owner == Owner.NONE
    illegal(happy_path(), ET.CASE_CANCELLED, {"reason": "x"})


# ------------------------------------------------------------------------------ patient track


def test_patient_consent_needs_both():
    log = opened().add(ET.ADDRESS_CONFIRMED, {"address": ADDR, "home_access": "ramp"})
    st = log.state()
    assert st.patient.state == PatientStatus.INTAKE and st.patient_info.profile.home_access == "ramp"
    st = log.add(ET.CONSENT_GIVEN, {"scope": "all"}).state()
    assert st.patient.state == PatientStatus.CONSENTED and st.patient.owner == Owner.US
    # reverse order works too
    st = opened().add(ET.CONSENT_GIVEN, {"scope": "x"}).add(ET.ADDRESS_CONFIRMED, {"address": ADDR}).state()
    assert st.patient.state == PatientStatus.CONSENTED


def test_patient_cost_flow():
    log = consented().add(ET.COST_DISCLOSED, {"cost": CostInfo()})
    st = log.state()
    assert st.patient.state == PatientStatus.COST_DISCLOSED and st.patient.owner == Owner.PATIENT
    assert st.patient.due_at == add_business_hours(log.t, 24)
    st = log.add(ET.COST_ACCEPTED).state()
    assert st.patient.state == PatientStatus.COST_ACCEPTED and st.patient.owner == Owner.US


def test_patient_illegal():
    illegal(opened(), ET.COST_DISCLOSED, {"cost": CostInfo()})  # not consented
    illegal(consented(), ET.COST_ACCEPTED)  # COST_ACCEPTED before COST_DISCLOSED


def test_concern_saves_and_restores_resume_to():
    log = consented().add(ET.COST_DISCLOSED, {"cost": CostInfo()})
    log.add(ET.CONCERN_RAISED, {"kind": "cost", "utterance": "too expensive"})
    e = esc("e1", Track.PATIENT, EscalationTrigger.COST_CONCERN, EscalationLane.PATIENT, ("ok", "decline"))
    log.add(ET.ESCALATION_OPENED, {"escalation": e})
    st = log.state()
    assert st.patient.state == PatientStatus.CONCERN_RAISED and st.resume_to == PatientStatus.COST_DISCLOSED
    assert st.patient.owner == Owner.ADVOCATE
    log.add(ET.HUMAN_RESOLVED, {"eid": "e1", "option_id": "ok", "resolver": "nurse"})
    st = log.state()
    assert st.patient.state == PatientStatus.COST_DISCLOSED and st.resume_to is None
    assert (
        st.escalations["e1"].status == EscalationStatus.RESOLVED and st.escalations["e1"].resolution == "ok"
    )
    st = log.add(ET.COST_ACCEPTED).state()  # effect event appended separately by the run loop
    assert st.patient.state == PatientStatus.COST_ACCEPTED


def test_patient_notified_counters():
    log = opened()
    log.add(
        ET.PATIENT_NOTIFIED,
        {"purpose": "welcome_intake", "text": "hi", "grounding_ok": True, "answered": False},
    )
    log.add(
        ET.PATIENT_NOTIFIED,
        {"purpose": "welcome_intake", "text": "hi", "grounding_ok": True, "answered": False},
    )
    st = log.state()
    assert st.notify_attempts == 2 and st.last_notified_at == log.t
    log.add(
        ET.PATIENT_NOTIFIED,
        {"purpose": "welcome_intake", "text": "hi", "grounding_ok": True, "answered": True},
    )
    assert log.state().notify_attempts == 0


# ------------------------------------------------------------------------------ order track


def test_order_happy_states_and_owners():
    log = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r1", "version": 1})
    st = log.state()
    assert (st.order.state, st.order.owner, st.order.due_at) == (OrderStatus.REQUESTED, Owner.PCP, None)
    due = SIM_START + timedelta(days=1)
    log.add(ET.COMMITMENT_MADE, {"cid": "c1", "party": "pcp", "what": "fax", "due_at": due})
    assert log.state().order.due_at == due
    log.add(ET.VISIT_NOTE_RECEIVED, {"note": VisitNote(doc_id="n1", text="t", received_at=log.t)})
    st = log.state()
    assert st.order.state == OrderStatus.DOCS_CHECK and st.order.owner == Owner.US
    assert st.commitments["c1"].status == CommitmentStatus.KEPT and st.order.due_at is None
    missing = {
        "checklist": LCDChecklist(results={LCDCriterion.MRADL_LIMITATION: "met"}),
        "missing": [c for c in LCDCriterion if c != LCDCriterion.MRADL_LIMITATION],
    }
    st = log.add(ET.LCD_EVALUATED, missing).state()
    assert st.order.state == OrderStatus.DOCS_CHECK and st.lcd is not None and not st.lcd.all_met
    st = log.add(ET.ADDENDUM_REQUESTED, {"missing": missing["missing"]}).state()
    assert st.order.state == OrderStatus.ADDENDUM_REQUESTED and st.addendum_requests == 1
    log.add(
        ET.VISIT_NOTE_RECEIVED,
        {"note": VisitNote(doc_id="n2", text="t", received_at=log.t, is_addendum=True)},
    )
    log.add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
    assert log.state().order.state == OrderStatus.DOCS_CHECK
    bad_swo = SIGNED_SWO.model_copy(update={"practitioner_npi": None})
    log.add(ET.SWO_SIGNED_RECEIVED, {"swo": bad_swo})
    log.add(ET.SWO_EVALUATED, {"complete": False, "missing": ["practitioner_npi"], "hcpcs": "K0001"})
    assert log.state().order.state == OrderStatus.SIGNED
    illegal(log, ET.ORDER_VALIDATED)  # SWO incomplete
    st = log.add(ET.CORRECTION_REQUESTED, {"fields": ["practitioner_npi"]}).state()
    assert st.order.state == OrderStatus.CORRECTION_REQUESTED and st.correction_requests == 1
    log.add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO})
    log.add(ET.SWO_EVALUATED, {"complete": True, "missing": [], "hcpcs": "K0001"})
    st = log.add(ET.ORDER_VALIDATED).state()
    assert st.order.state == OrderStatus.VALID


def test_order_validated_guards():
    base = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    base.add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO})
    illegal(base, ET.ORDER_VALIDATED)  # no LCD
    drift = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    drift.add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
    drift.add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO.model_copy(update={"hcpcs": "K0003"})})
    illegal(drift, ET.ORDER_VALIDATED)  # code drift


def test_order_illegal_and_resend():
    log = consented()
    illegal(log, ET.ORDER_REQUESTED, {"request_ref": "r", "version": 0})
    log.add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    illegal(log, ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})  # duplicate version
    log.add(ET.ORDER_REQUEST_VERIFIED, {"received": False})
    st = log.add(ET.ORDER_REQUESTED, {"request_ref": "r2", "version": 2}).state()
    assert st.order_request_version == 2 and st.order.state == OrderStatus.REQUESTED
    illegal(log, ET.ORDER_VALIDATED)  # not SIGNED
    illegal(log, ET.CORRECTION_REQUESTED, {"fields": ["x"]})
    illegal(opened(), ET.ADDENDUM_REQUESTED, {"missing": []})


def test_order_sent_must_target_committed():
    log = committed(order_valid(consented()))
    illegal(log, ET.ORDER_SENT, {"to_sid": B})  # not committed
    log.add(ET.ORDER_SENT, {"to_sid": A})
    st = log.state()
    assert (
        st.order.state == OrderStatus.SENT_TO and st.order.owner == Owner.SUPPLIER and st.order_sent_to == A
    )
    illegal(log, ET.ORDER_RECEIVED, {"by_sid": B})
    st = log.add(ET.ORDER_RECEIVED, {"by_sid": A}).state()
    assert st.order.state == OrderStatus.RECEIVED_BY and st.order_received_by == A
    illegal(order_valid(consented()), ET.ORDER_SENT, {"to_sid": A})  # nobody committed


def test_order_redirect_back_to_valid():
    log = (
        committed(order_valid(consented()))
        .add(ET.ORDER_SENT, {"to_sid": A})
        .add(ET.ORDER_RECEIVED, {"by_sid": A})
    )
    log.add(ET.SUPPLIER_RELEASED, {"sid": A, "reason": "ghosted"})
    st = log.add(ET.ORDER_REDIRECT, {"from_sid": A, "to_sid": B}).state()
    assert st.order.state == OrderStatus.VALID and st.order_sent_to is None and st.order_received_by is None
    assert st.committed_sid is None and st.supplier.state == SupplierTrackStatus.QUALIFIED
    assert st.supplier.party_id == B
    log.add(ET.SUPPLIER_COMMITTED, {"sid": B, "ref": "R2", "expires_at": log.t + timedelta(hours=72)})
    st = log.add(ET.ORDER_SENT, {"to_sid": B}).state()
    assert st.order_sent_to == B
    illegal(order_valid(consented()), ET.ORDER_REDIRECT, {"from_sid": A, "to_sid": B})


@pytest.mark.parametrize(
    "blocker",
    [
        (ET.ORDER_STALLED, {"followups": 3}),
        (ET.PCP_REFUSED, {"reason": "clinical question"}),
    ],
)
def test_order_blocked_and_unblocked(blocker):
    log = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1}).add(*blocker)
    st = log.state()
    assert st.order.state == OrderStatus.BLOCKED_ON_HUMAN and st.order.owner == Owner.ADVOCATE
    log.add(ET.ESCALATION_OPENED, {"escalation": esc("e1", Track.ORDER, EscalationTrigger.ORDER_STALLED)})
    st = log.add(ET.HUMAN_RESOLVED, {"eid": "e1", "option_id": "a", "resolver": "adv"}).state()
    assert st.order.state == OrderStatus.REQUESTED and st.order.owner == Owner.PCP


def test_escalation_on_order_blocks_and_code_drift_fix():
    log = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    log.add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
    log.add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO.model_copy(update={"hcpcs": "K0003"})})
    log.add(
        ET.ESCALATION_OPENED,
        {"escalation": esc("d1", Track.ORDER, EscalationTrigger.CODE_DRIFT, opts=("fix",))},
    )
    assert log.state().order.state == OrderStatus.BLOCKED_ON_HUMAN
    illegal(log, ET.HUMAN_RESOLVED, {"eid": "d1", "option_id": "nope", "resolver": "adv"})
    illegal(log, ET.HUMAN_RESOLVED, {"eid": "zz", "option_id": "fix", "resolver": "adv"})
    log.add(ET.HUMAN_RESOLVED, {"eid": "d1", "option_id": "fix", "resolver": "adv"})
    assert log.state().order.state == OrderStatus.SIGNED
    log.add(ET.SWO_SIGNED_RECEIVED, {"swo": SIGNED_SWO})
    st = log.add(ET.ORDER_VALIDATED).state()
    assert st.order.state == OrderStatus.VALID
    illegal(log, ET.HUMAN_RESOLVED, {"eid": "d1", "option_id": "fix", "resolver": "adv"})  # already resolved
    illegal(log, ET.ESCALATION_OPENED, {"escalation": esc("d1", Track.ORDER, EscalationTrigger.CODE_DRIFT)})


# ------------------------------------------------------------------------------ supplier + assignment


def test_supplier_records_and_derived_track():
    log = consented().add(ET.CANDIDATES_RANKED, {"sids": [A, B], "distances": {A: 1.0, B: 2.0}})
    st = log.state()
    assert st.supplier.state == SupplierTrackStatus.CALLING and st.ranked == [A, B]
    assert st.suppliers[A].status == SupplierStatus.UNCONTACTED
    for i in (1, 2):
        log.add(ET.SUPPLIER_NO_ANSWER, {"sid": A, "attempt": i})
    st = log.state()
    assert st.suppliers[A].status == SupplierStatus.NO_ANSWER and st.suppliers[A].attempts == 2
    st = log.add(ET.SUPPLIER_NO_ANSWER, {"sid": A, "attempt": 3}).state()
    assert st.suppliers[A].status == SupplierStatus.UNREACHABLE
    assert st.supplier.state == SupplierTrackStatus.CALLING and st.supplier.party_id == B
    st = log.add(ET.SUPPLIER_DISQUALIFIED, {"sid": B, "reason": "out of stock"}).state()
    assert st.supplier.state == SupplierTrackStatus.EXHAUSTED and st.suppliers[B].reason == "out of stock"
    illegal(log, ET.SUPPLIER_NO_ANSWER, {"sid": "unknown", "attempt": 1})


def test_supplier_qualify_commit_backup():
    log = committed(consented())
    st = log.state()
    assert st.committed_sid == A and st.backup_sid == B
    assert st.supplier.state == SupplierTrackStatus.COMMITTED and st.supplier.owner == Owner.SUPPLIER
    assert st.supplier.due_at == st.suppliers[A].expires_at
    assert st.assignment.state == AssignmentStatus.CONFIRMED_ALL_MONTHS and st.assignment.party_id == A
    # second commit while one is COMMITTED
    illegal(log, ET.SUPPLIER_COMMITTED, {"sid": B, "ref": "x", "expires_at": log.t})
    illegal(log, ET.SUPPLIER_COMMITTED, {"sid": A, "ref": "x", "expires_at": log.t})
    illegal(log, ET.SUPPLIER_DISQUALIFIED, {"sid": A, "reason": "x"})  # must release first
    t2 = log.t + timedelta(hours=1)
    st = log.add(ET.SUPPLIER_REVERIFIED, {"sid": A, "verified_at": t2}).state()
    assert st.suppliers[A].verified_at == t2
    st = log.add(ET.SUPPLIER_RELEASED, {"sid": A, "reason": "lease expired"}).state()
    assert st.committed_sid is None and st.suppliers[A].status == SupplierStatus.RELEASED
    assert st.supplier.state == SupplierTrackStatus.QUALIFIED and st.supplier.party_id == B
    # released supplier is never re-qualified
    st = log.add(ET.SUPPLIER_QUALIFIED, {"sid": A, "facts": GOOD_FACTS, "verified_at": log.t}).state()
    assert st.suppliers[A].status == SupplierStatus.RELEASED


def test_commit_requires_assignment_and_qualified():
    log = consented().add(ET.CANDIDATES_RANKED, {"sids": [A, B], "distances": {A: 1.0, B: 2.0}})
    illegal(log, ET.SUPPLIER_COMMITTED, {"sid": A, "ref": "x", "expires_at": log.t})  # not qualified
    no_assign = GOOD_FACTS.model_copy(update={"accepts_assignment_all_months": None})
    log.add(ET.SUPPLIER_QUALIFIED, {"sid": A, "facts": no_assign, "verified_at": log.t})
    illegal(log, ET.SUPPLIER_COMMITTED, {"sid": A, "ref": "x", "expires_at": log.t})  # assignment unknown
    log.add(ET.ASSIGNMENT_REFUSED, {"sid": A})
    assert log.state().assignment.state == AssignmentStatus.REFUSED
    illegal(log, ET.SUPPLIER_COMMITTED, {"sid": A, "ref": "x", "expires_at": log.t})
    log.add(ET.ASSIGNMENT_CONFIRMED, {"sid": A})
    st = log.add(ET.SUPPLIER_COMMITTED, {"sid": A, "ref": "x", "expires_at": log.t}).state()
    assert st.committed_sid == A
    illegal(log, ET.ASSIGNMENT_CONFIRMED, {"sid": "nobody"})


def test_best_qualified_prefers_assignment_then_rank():
    log = consented().add(ET.CANDIDATES_RANKED, {"sids": [A, B], "distances": {A: 1.0, B: 2.0}})
    unknown = GOOD_FACTS.model_copy(update={"accepts_assignment_all_months": None})
    log.add(ET.SUPPLIER_QUALIFIED, {"sid": A, "facts": unknown, "verified_at": log.t})
    log.add(ET.SUPPLIER_QUALIFIED, {"sid": B, "facts": GOOD_FACTS, "verified_at": log.t})
    st = log.state()
    assert st.supplier.party_id == B and st.backup_sid == A


# ------------------------------------------------------------------------------ delivery


def test_delivery_happy_path():
    log = ready_to_schedule()
    st = log.state()
    assert st.delivery.state == DeliveryStatus.SCHEDULING and st.delivery.owner == Owner.PATIENT
    slot = log.t + timedelta(days=2)
    st = log.add(ET.DELIVERY_SCHEDULED, {"sid": A, "slot": slot}).state()
    assert st.delivery.state == DeliveryStatus.SCHEDULED and st.delivery.due_at == slot + timedelta(hours=4)
    assert st.delivery.owner == Owner.SUPPLIER
    st = log.add(ET.DELIVERY_MISSED).state()
    assert st.delivery.state == DeliveryStatus.SCHEDULING
    st = happy_path().state()
    assert st.delivery.state == DeliveryStatus.POD_CONFIRMED
    assert st.patient.state == PatientStatus.DELIVERY_CONFIRMED


@pytest.mark.parametrize(
    "mutate",
    [
        "no_cost_accept",
        "stale_verify",
        "not_received",
        "wrong_sid",
    ],
)
def test_delivery_scheduled_guard(mutate):
    log = consented()
    order_valid(log)
    committed(log)
    log.add(ET.COST_DISCLOSED, {"cost": CostInfo()})
    if mutate != "no_cost_accept":
        log.add(ET.COST_ACCEPTED)
    log.add(ET.ORDER_SENT, {"to_sid": A})
    if mutate != "not_received":
        log.add(ET.ORDER_RECEIVED, {"by_sid": A})
    log.add(ET.DELIVERY_SLOT_PROPOSED, {"sid": A, "slot": log.t + timedelta(days=2)})
    if mutate == "stale_verify":
        log.t += timedelta(hours=25)
    sid = B if mutate == "wrong_sid" else A
    illegal(log, ET.DELIVERY_SCHEDULED, {"sid": sid, "slot": log.t + timedelta(days=2)})


def test_delivery_illegal_order():
    illegal(opened(), ET.DELIVERED, {"sid": A, "at": SIM_START})
    illegal(opened(), ET.POD_CONFIRMED)
    illegal(opened(), ET.DELIVERY_MISSED)
    illegal(opened(), ET.DELIVERY_SCHEDULED, {"sid": A, "slot": SIM_START})
    illegal(consented(), ET.DELIVERY_SLOT_PROPOSED, {"sid": A, "slot": SIM_START})  # nobody committed


def test_release_of_committed_resets_delivery():
    log = ready_to_schedule()
    st = log.add(ET.SUPPLIER_RELEASED, {"sid": A, "reason": "ghost"}).state()
    assert st.delivery.state == DeliveryStatus.NOT_STARTED and st.delivery_slot is None


# ------------------------------------------------------------------------------ ledger + escalation owner


def test_ledger():
    log = opened()
    due = SIM_START + timedelta(days=1)
    log.add(ET.COMMITMENT_MADE, {"cid": "k1", "party": A, "what": "call back", "due_at": due})
    log.add(ET.COMMITMENT_MADE, {"cid": "k2", "party": "pcp", "what": "fax", "due_at": due})
    st = log.state()
    assert st.commitments["k1"].status == CommitmentStatus.OPEN
    assert st.commitments["k1"].source_event_id == log.events[1].event_id
    st = log.add(ET.COMMITMENT_BROKEN, {"cid": "k1"}).state()
    assert st.commitments["k1"].status == CommitmentStatus.BROKEN and st.commitments["k1"].followups == 1
    st = log.add(ET.COMMITMENT_KEPT, {"cid": "k2"}).state()
    assert st.commitments["k2"].status == CommitmentStatus.KEPT
    illegal(log, ET.COMMITMENT_KEPT, {"cid": "nope"})
    illegal(log, ET.COMMITMENT_BROKEN, {"cid": "nope"})


def test_supplier_commitment_closed_by_delivery():
    log = ready_to_schedule()
    log.add(
        ET.COMMITMENT_MADE, {"cid": "s1", "party": A, "what": "deliver", "due_at": log.t + timedelta(days=3)}
    )
    slot = log.t + timedelta(days=2)
    log.add(ET.DELIVERY_SCHEDULED, {"sid": A, "slot": slot}).add(ET.DELIVERED, {"sid": A, "at": slot})
    assert log.state().commitments["s1"].status == CommitmentStatus.KEPT


def test_escalation_sets_and_restores_owner():
    log = consented().add(ET.CANDIDATES_RANKED, {"sids": [A], "distances": {A: 1.0}})
    log.add(ET.SUPPLIER_DISQUALIFIED, {"sid": A, "reason": "x"})
    e = esc("x1", Track.SUPPLIER, EscalationTrigger.SUPPLIERS_EXHAUSTED, opts=("expand_radius",))
    st = log.add(ET.ESCALATION_OPENED, {"escalation": e}).state()
    assert st.supplier.owner == Owner.ADVOCATE and st.patient.owner == Owner.US  # only that track blocks
    st = log.add(ET.HUMAN_RESOLVED, {"eid": "x1", "option_id": "expand_radius", "resolver": "adv"}).state()
    assert st.supplier.owner == Owner.US and st.supplier.state == SupplierTrackStatus.EXHAUSTED



def test_order_request_verified_not_received_resets_to_not_sent():
    """Orchestrator fix: the planner relies on 'we never got it' -> NOT_SENT for a versioned resend."""
    log = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r1", "version": 1})
    from datetime import timedelta as _td
    log.add(ET.COMMITMENT_MADE, {"cid": "c1", "party": "pcp", "what": "fax order",
                                 "due_at": log.t + _td(days=1), "evidence": "will fax"})
    assert log.state().order.state == OrderStatus.REQUESTED
    st = log.add(ET.ORDER_REQUEST_VERIFIED, {"received": False}).state()
    assert st.order.state == OrderStatus.NOT_SENT
    assert st.commitments["c1"].status == "BROKEN"
    st_ok = consented().add(ET.ORDER_REQUESTED, {"request_ref": "r1", "version": 1}).add(
        ET.ORDER_REQUEST_VERIFIED, {"received": True}).state()
    assert st_ok.order.state == OrderStatus.REQUESTED
