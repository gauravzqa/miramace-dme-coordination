from datetime import date, datetime, timedelta

from miramace.clock import SIM_START, add_business_hours, is_business
from miramace.events import EventType
from miramace.models import (
    CHICAGO,
    SWO,
    Address,
    AssignmentStatus,
    CaseState,
    CaseStatus,
    Commitment,
    DeliveryStatus,
    Escalation,
    EscalationTrigger,
    LCDChecklist,
    LCDCriterion,
    OrderStatus,
    Owner,
    PatientStatus,
    SupplierRecord,
    SupplierStatus,
    SupplierTrackStatus,
    TaskKind,
    TrackState,
    dr_chen,
    eleanor,
)
from miramace.planner import Policy, plan

LAKE = "lakeshore-home-medical-equipment"
LINC = "lincoln-park-dme-services"
ADDR = Address.parse("2100 N Lincoln Park W, Chicago, IL 60614")


def patient_with_address():
    return eleanor().model_copy(update={"address": ADDR})


def consented(**kw) -> CaseState:
    base = dict(
        case_id="c1",
        patient=TrackState(state=PatientStatus.CONSENTED, owner=Owner.US),
        patient_info=patient_with_address(),
        pcp=dr_chen(),
        patient_flags={"address", "consent"},
    )
    base.update(kw)
    return CaseState(**base)


def esc_triggers(dec):
    return [e.parsed().escalation.trigger for e in dec.events if e.type == EventType.ESCALATION_OPENED]


def by_intent(dec):
    return {(t.party_id, t.intent): t for t in dec.tasks}


def full_swo(**kw):
    d = dict(
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
    d.update(kw)
    return SWO(**d)


def test_fresh_consented_requests_order_and_screens_nearest_only():
    dec = plan(consented(), SIM_START)
    tasks = by_intent(dec)
    assert set(tasks) == {("pcp", "request_order"), (LAKE, "screen")}
    ro = tasks[("pcp", "request_order")]
    assert ro.kind == TaskKind.CALL_PCP and ro.version == 1 and ro.idem_key == "c1:pcp:request_order:v1"
    sc = tasks[(LAKE, "screen")]
    assert sc.kind == TaskKind.CALL_SUPPLIER and sc.due_at == SIM_START
    # no PHI in screening args
    blob = str(sc.args)
    assert (
        "Eleanor" not in blob and "1EG4" not in blob and "1954" not in blob and "Lincoln Park W" not in blob
    )
    assert [e.type for e in dec.events] == [EventType.CANDIDATES_RANKED]
    assert dec.events[0].source.kind == "RULE"
    assert dec.events[0].parsed().sids[:2] == [LAKE, LINC]


def test_plan_is_idempotent():
    s = consented()
    assert plan(s, SIM_START) == plan(s, SIM_START)
    s2 = consented(
        patient=TrackState(state=PatientStatus.INTAKE, owner=Owner.PATIENT),
        patient_flags=set(),
        patient_info=eleanor(),
    )
    assert plan(s2, SIM_START) == plan(s2, SIM_START)


def test_one_screen_in_flight_and_stop_at_two_qualified():
    ranked = [LAKE, LINC, "halsted-home-health-supply"]
    s = consented(
        ranked=ranked,
        suppliers={
            LAKE: SupplierRecord(sid=LAKE, status=SupplierStatus.DISQUALIFIED, attempts=1),
            LINC: SupplierRecord(sid=LINC, status=SupplierStatus.UNCONTACTED),
        },
    )
    screens = [t for t in plan(s, SIM_START).tasks if t.intent == "screen"]
    assert [t.party_id for t in screens] == [LINC]
    assert not any(e.type == EventType.CANDIDATES_RANKED for e in plan(s, SIM_START).events)

    s = consented(
        ranked=ranked,
        suppliers={
            LAKE: SupplierRecord(sid=LAKE, status=SupplierStatus.QUALIFIED, attempts=1),
            LINC: SupplierRecord(sid=LINC, status=SupplierStatus.QUALIFIED, attempts=1),
        },
    )
    assert not [t for t in plan(s, SIM_START).tasks if t.intent == "screen"]


def test_no_answer_retry_and_exhaustion():
    ranked = [LAKE]
    s = consented(
        ranked=ranked, suppliers={LAKE: SupplierRecord(sid=LAKE, status=SupplierStatus.NO_ANSWER, attempts=1)}
    )
    t = [t for t in plan(s, SIM_START).tasks if t.intent == "screen"][0]
    assert t.version == 2 and t.due_at == add_business_hours(SIM_START, 2)
    s = consented(
        ranked=ranked,
        suppliers={LAKE: SupplierRecord(sid=LAKE, status=SupplierStatus.UNREACHABLE, attempts=3)},
    )
    dec = plan(s, SIM_START)
    escs = [e for e in dec.events if e.type == EventType.ESCALATION_OPENED]
    assert len(escs) == 1 and escs[0].parsed().escalation.trigger == EscalationTrigger.SUPPLIERS_EXHAUSTED


def test_followups_at_24_48_72_bh_then_stalled():
    due = datetime(2026, 11, 3, 10, tzinfo=CHICAGO)
    for k, offs in [(1, 24), (2, 48), (3, 72)]:
        cm = Commitment(cid="cm1", party="pcp", what="fax visit note + SWO", due_at=due, followups=k - 1)
        s = consented(
            order=TrackState(state=OrderStatus.REQUESTED, owner=Owner.PCP),
            order_request_version=1,
            commitments={"cm1": cm},
        )
        t = by_intent(plan(s, due))[("pcp", f"followup_order_{k}")]
        assert t.due_at == add_business_hours(due, offs)
        assert is_business(t.due_at)
    assert add_business_hours(due, 24) == datetime(2026, 11, 6, 10, tzinfo=CHICAGO)
    cm = Commitment(cid="cm1", party="pcp", what="fax", due_at=due, followups=3)
    s = consented(order=TrackState(state=OrderStatus.REQUESTED, owner=Owner.PCP), commitments={"cm1": cm})
    dec = plan(s, due + timedelta(days=10))
    assert not [t for t in dec.tasks if t.party_id == "pcp"]
    esc = [e for e in dec.events if e.type == EventType.ESCALATION_OPENED][0].parsed().escalation
    assert esc.trigger == EscalationTrigger.ORDER_STALLED and esc.track == "order"


def test_never_reemit_open_escalation():
    esc = Escalation(
        eid="esc-order_stalled-1",
        lane="CASE",
        trigger=EscalationTrigger.ORDER_STALLED,
        track="order",
        question="?",
        options=[],
        context={"key": "cm1"},
    )
    cm = Commitment(cid="cm1", party="pcp", what="fax", due_at=SIM_START, followups=3)
    s = consented(
        order=TrackState(state=OrderStatus.REQUESTED, owner=Owner.PCP),
        commitments={"cm1": cm},
        escalations={esc.eid: esc},
    )
    assert not [e for e in plan(s, SIM_START).events if e.type == EventType.ESCALATION_OPENED]


def test_advocate_owned_track_gets_no_tasks():
    s = consented(order=TrackState(state=OrderStatus.NOT_SENT, owner=Owner.ADVOCATE))
    dec = plan(s, SIM_START)
    assert not [t for t in dec.tasks if t.kind == TaskKind.CALL_PCP]
    assert [t for t in dec.tasks if t.intent == "screen"]  # other tracks continue
    s = consented(supplier=TrackState(state=SupplierTrackStatus.CALLING, owner=Owner.ADVOCATE))
    assert not [t for t in plan(s, SIM_START).tasks if t.kind == TaskKind.CALL_SUPPLIER]


def test_not_active_plans_nothing():
    s = consented(case=TrackState(state=CaseStatus.PAUSED))
    dec = plan(s, SIM_START)
    assert dec.tasks == [] and dec.events == []


def test_welcome_ladder_and_unreachable():
    base = dict(
        patient=TrackState(state=PatientStatus.INTAKE, owner=Owner.PATIENT),
        patient_flags=set(),
        patient_info=eleanor(),
    )
    night = datetime(2026, 11, 2, 22, tzinfo=CHICAGO)
    t = by_intent(plan(consented(**base), night))[("patient", "welcome_intake")]
    assert t.version == 1 and t.due_at == datetime(2026, 11, 3, 9, tzinfo=CHICAGO)
    s = consented(**base, notify_attempts=3, last_notified_at=SIM_START)
    dec = plan(s, SIM_START + timedelta(hours=5))
    assert not [t for t in dec.tasks if t.kind == TaskKind.NOTIFY_PATIENT]
    esc = [e for e in dec.events if e.type == EventType.ESCALATION_OPENED][0].parsed().escalation
    assert esc.trigger == EscalationTrigger.PATIENT_UNREACHABLE and esc.lane == "PATIENT"
    prof = eleanor().profile.model_copy(update={"caregiver_phone": "(312) 555-0100"})
    s = consented(
        **{**base, "patient_info": eleanor().model_copy(update={"profile": prof})},
        notify_attempts=3,
        last_notified_at=SIM_START,
    )
    t = by_intent(plan(s, SIM_START + timedelta(hours=5)))[("patient", "welcome_intake")]
    assert t.version == 4 and t.args["to"] == "caregiver"


def _all_met():
    return LCDChecklist(results={c: "met" for c in LCDCriterion})


def test_lcd_missing_addendum_then_escalate():
    lcd = LCDChecklist(results={c: "met" for c in LCDCriterion if c != LCDCriterion.HOME_ACCOMMODATES})
    s = consented(order=TrackState(state=OrderStatus.DOCS_CHECK), lcd=lcd, order_request_version=1)
    t = by_intent(plan(s, SIM_START))[("pcp", "request_addendum")]
    assert t.args["missing"] == ["HOME_ACCOMMODATES"] and t.args["wording"]
    s = s.model_copy(update={"addendum_requests": 1})
    dec = plan(s, SIM_START)
    assert ("pcp", "request_addendum") not in by_intent(dec)
    assert esc_triggers(dec)[-1] == EscalationTrigger.LCD_MISSING_AFTER_ADDENDUM


def test_signed_code_drift_correction_and_validation():
    signed = dict(order=TrackState(state=OrderStatus.SIGNED), lcd=_all_met(), order_request_version=1)
    dec = plan(consented(**signed, swo=full_swo(hcpcs="K0003")), SIM_START)
    assert [e.parsed().escalation.trigger for e in dec.events if e.type == EventType.ESCALATION_OPENED] == [
        EscalationTrigger.CODE_DRIFT
    ]
    dec = plan(consented(**signed, swo=full_swo(practitioner_npi=None)), SIM_START)
    assert by_intent(dec)[("pcp", "request_correction")].args["fields"] == ["practitioner_npi"]
    dec = plan(consented(**signed, swo=full_swo(practitioner_npi=None), correction_requests=1), SIM_START)
    assert esc_triggers(dec)[-1] == EscalationTrigger.SWO_INCOMPLETE_AFTER_CORRECTION
    dec = plan(consented(**signed, swo=full_swo()), SIM_START)
    assert EventType.ORDER_VALIDATED in [e.type for e in dec.events]


def committed_state(**kw):
    base = dict(
        ranked=[LAKE, LINC],
        committed_sid=LAKE,
        supplier=TrackState(state=SupplierTrackStatus.COMMITTED, owner=Owner.SUPPLIER),
        suppliers={
            LAKE: SupplierRecord(
                sid=LAKE,
                status=SupplierStatus.COMMITTED,
                verified_at=SIM_START,
                expires_at=SIM_START + timedelta(hours=72),
                ref="R1",
            ),
            LINC: SupplierRecord(sid=LINC, status=SupplierStatus.QUALIFIED, verified_at=SIM_START),
        },
        assignment_by_sid={
            LAKE: AssignmentStatus.CONFIRMED_ALL_MONTHS,
            LINC: AssignmentStatus.CONFIRMED_ALL_MONTHS,
        },
        order=TrackState(state=OrderStatus.VALID),
        order_request_version=1,
        swo=full_swo(),
        lcd=_all_met(),
    )
    base.update(kw)
    return consented(**base)


def test_commit_cost_and_send_order():
    s = consented(
        ranked=[LAKE, LINC],
        suppliers={
            LAKE: SupplierRecord(sid=LAKE, status=SupplierStatus.QUALIFIED, verified_at=SIM_START),
            LINC: SupplierRecord(sid=LINC, status=SupplierStatus.QUALIFIED, verified_at=SIM_START),
        },
        assignment_by_sid={LAKE: AssignmentStatus.CONFIRMED_ALL_MONTHS},
    )
    assert (LAKE, "commit") in by_intent(plan(s, SIM_START))
    tasks = by_intent(plan(committed_state(), SIM_START))
    assert ("patient", "cost_disclosure") in tasks
    assert tasks[("pcp", f"send_order_{LAKE}")].args == {"to_sid": LAKE}
    assert (LAKE, "reverify") in tasks  # lease renewal timer
    assert tasks[(LAKE, "reverify")].due_at < SIM_START + timedelta(hours=72)


def test_lease_expiry_releases_and_redirects():
    s = committed_state(order=TrackState(state=OrderStatus.SENT_TO, owner=Owner.SUPPLIER), order_sent_to=LAKE)
    later = SIM_START + timedelta(hours=73)
    dec = plan(s, later)
    types = [e.type for e in dec.events]
    assert types == [EventType.SUPPLIER_RELEASED, EventType.ORDER_REDIRECT]
    assert dec.events[1].parsed().to_sid == LINC
    assert not [t for t in dec.tasks if t.kind != TaskKind.NOTIFY_PATIENT]
    assert plan(s, later) == dec


def test_schedule_flow_and_done():
    s = committed_state(
        patient=TrackState(state=PatientStatus.COST_ACCEPTED),
        order=TrackState(state=OrderStatus.RECEIVED_BY),
        order_sent_to=LAKE,
        order_received_by=LAKE,
    )
    assert (LAKE, "propose_slot") in by_intent(plan(s, SIM_START + timedelta(hours=1)))
    t = by_intent(plan(s, SIM_START + timedelta(hours=30)))[(LAKE, "reverify")]
    assert t.args["reason"] == "pre_schedule"
    slot = SIM_START + timedelta(days=2)
    s2 = s.model_copy(
        update={
            "delivery": TrackState(state=DeliveryStatus.SCHEDULING, owner=Owner.PATIENT),
            "delivery_slot": slot,
        }
    )
    assert ("patient", "delivery_slot") in by_intent(plan(s2, SIM_START + timedelta(hours=1)))
    s3 = s.model_copy(
        update={
            "delivery": TrackState(state=DeliveryStatus.SCHEDULED, due_at=slot + timedelta(hours=4)),
            "delivery_slot": slot,
        }
    )
    assert by_intent(plan(s3, SIM_START))[(LAKE, "verify_delivered")].due_at == slot + timedelta(hours=4)
    s4 = s.model_copy(
        update={"delivery": TrackState(state=DeliveryStatus.DELIVERED), "last_notified_at": SIM_START}
    )
    t = by_intent(plan(s4, SIM_START + timedelta(hours=1)))[("patient", "delivered_checkin")]
    assert t.due_at == datetime(2026, 11, 3, 9, tzinfo=CHICAGO)  # capped to 1/day
    s5 = s.model_copy(update={"delivery": TrackState(state=DeliveryStatus.POD_CONFIRMED)})
    assert [e.type for e in plan(s5, SIM_START).events] == [EventType.CASE_DONE]


def test_policy_default_is_frozen():
    assert Policy().lease_h == 72 and Policy().pcp_followup_offsets_bh == (24, 48, 72)
