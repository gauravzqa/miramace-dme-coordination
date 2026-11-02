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
    VisitNote,
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


def test_followups_at_8_16_24_bh_then_stalled():
    due = datetime(2026, 11, 3, 10, tzinfo=CHICAGO)
    for k, offs in [(1, 8), (2, 16), (3, 24)]:
        cm = Commitment(cid="cm1", party="pcp", what="fax visit note + SWO", due_at=due, followups=k - 1)
        s = consented(
            order=TrackState(state=OrderStatus.REQUESTED, owner=Owner.PCP),
            order_request_version=1,
            commitments={"cm1": cm},
        )
        t = by_intent(plan(s, due))[("pcp", f"followup_order_{k}")]
        assert t.due_at == add_business_hours(due, offs)
        assert is_business(t.due_at)
    assert add_business_hours(due, 8) == datetime(2026, 11, 4, 10, tzinfo=CHICAGO)
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
    # ladder position = sends of this purpose (notify_counts), so answered-but-incomplete replies re-ask too
    s = consented(**base, notify_counts={"welcome_intake": 3}, last_notified_at=SIM_START)
    dec = plan(s, SIM_START + timedelta(hours=5))
    assert not [t for t in dec.tasks if t.kind == TaskKind.NOTIFY_PATIENT]
    esc = [e for e in dec.events if e.type == EventType.ESCALATION_OPENED][0].parsed().escalation
    assert esc.trigger == EscalationTrigger.PATIENT_UNREACHABLE and esc.lane == "PATIENT"
    prof = eleanor().profile.model_copy(update={"caregiver_phone": "(312) 555-0100"})
    s = consented(
        **{**base, "patient_info": eleanor().model_copy(update={"profile": prof})},
        notify_counts={"welcome_intake": 3},
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
    s = s.model_copy(update={"addendum_requests": 1, "addendum_notes_seen": 0})
    dec = plan(s, SIM_START)  # requested, no note since: wait (no re-request, no escalation)
    assert ("pcp", "request_addendum") not in by_intent(dec) and not esc_triggers(dec)
    note = VisitNote(doc_id="add-1", text="addendum", received_at=SIM_START, is_addendum=True)
    s = s.model_copy(update={"visit_notes": [note]})  # a note arrived AFTER the request; gap remains
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
    rev = sorted(t.due_at for t in plan(committed_state(), SIM_START).tasks if t.intent == "reverify")
    end = SIM_START + timedelta(hours=72)
    assert len(rev) == 2 and rev[0] < end <= rev[1]  # renewal before lease end + wake-up at lease end


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
    assert Policy().lease_h == 72 and Policy().pcp_followup_offsets_bh == (8, 16, 24)


# ------------------------------------------------------------------ F1: liveness fixes (event-driven)
from miramace.events import Source, make_event  # noqa: E402
from miramace.models import CostInfo, SupplierFacts  # noqa: E402
from miramace.reducer import initial_state, reduce  # noqa: E402

_ET = EventType
_GOOD = SupplierFacts(
    taking_new_medicare=True, k0001_in_stock=True, delivers_to_zip=True, accepts_assignment_all_months=True
)


class _Log:
    def __init__(self) -> None:
        self.now = SIM_START
        self.state = initial_state("c1", eleanor(), dr_chen())

    def add(self, t, payload=None):
        e = make_event("c1", t, payload, Source(kind="SYSTEM", ref="t"), self.now)
        self.state = reduce(self.state, e.model_copy(update={"seq": self.state.last_seq + 1}))
        return self

    def apply_plan(self):
        dec = plan(self.state, self.now)
        for e in dec.events:
            self.state = reduce(self.state, e.model_copy(update={"seq": self.state.last_seq + 1}))
        return dec


def _committed_log() -> _Log:
    g = _Log()
    g.add(_ET.ADDRESS_CONFIRMED, {"address": ADDR}).add(_ET.CONSENT_GIVEN, {"scope": "x"})
    g.apply_plan()
    a, b = g.state.ranked[:2]
    g.add(_ET.SUPPLIER_QUALIFIED, {"sid": a, "facts": _GOOD, "verified_at": g.now})
    g.add(_ET.SUPPLIER_QUALIFIED, {"sid": b, "facts": _GOOD, "verified_at": g.now})
    g.add(_ET.SUPPLIER_COMMITTED, {"sid": a, "ref": "r", "expires_at": g.now + timedelta(hours=72)})
    return g


def test_two_unanswered_reverifies_release_committed_supplier():
    g = _committed_log()
    a = g.state.committed_sid
    v0 = {t.idem_key for t in plan(g.state, g.now).tasks if t.intent == "reverify"}
    g.add(_ET.SUPPLIER_NO_ANSWER, {"sid": a, "attempt": 1})
    assert g.state.suppliers[a].status == SupplierStatus.COMMITTED and g.state.suppliers[a].attempts == 1
    v1 = {t.idem_key for t in plan(g.state, g.now).tasks if t.intent == "reverify"}
    assert v1 - v0  # retry gets a fresh key
    g.add(_ET.SUPPLIER_NO_ANSWER, {"sid": a, "attempt": 2})
    dec = g.apply_plan()
    assert [e.type for e in dec.events] == [_ET.SUPPLIER_RELEASED]
    assert g.state.committed_sid is None


def test_verify_order_received_replanned_after_no_answer():
    g = _committed_log()
    a = g.state.committed_sid
    s = g.state.model_copy(
        update={"order": TrackState(state=OrderStatus.SENT_TO, owner=Owner.SUPPLIER), "order_sent_to": a}
    )

    def keys():
        return {(t.party_id, t.intent, t.version) for t in plan(s, g.now).tasks}

    assert (a, "verify_order_received", 1) in keys()
    s.suppliers[a].attempts = 1
    assert (a, "verify_order_received", 2) in keys()


def test_cost_disclosure_ladder_then_unreachable():
    g = _committed_log()
    for i in range(3):
        g.add(
            _ET.PATIENT_NOTIFIED,
            {"purpose": "cost_disclosure", "text": "x", "grounding_ok": True, "answered": False},
        )
        g.add(_ET.COST_DISCLOSED, {"cost": CostInfo()})
        t = [t for t in plan(g.state, g.now).tasks if t.intent == "cost_disclosure"]
        if i < 2:
            assert t and t[0].version == i + 2 and t[0].due_at >= g.state.patient.due_at
    dec = plan(g.state, g.now)
    assert esc_triggers(dec) == [EscalationTrigger.PATIENT_UNREACHABLE]


def test_patient_declines_slot_reschedule_reproposes():
    g = _committed_log()
    a = g.state.committed_sid
    g.add(_ET.DELIVERY_SLOT_PROPOSED, {"sid": a, "slot": g.now + timedelta(days=2)})
    g.add(_ET.DELIVERY_MISSED)  # `reschedule` intent while a slot is proposed
    assert g.state.delivery.state == DeliveryStatus.SCHEDULING and g.state.delivery_slot is None
    s = g.state.model_copy(
        update={
            "patient": TrackState(state=PatientStatus.COST_ACCEPTED),
            "patient_flags": g.state.patient_flags | {"cost_disclosed", "cost_accepted"},
            "order": TrackState(state=OrderStatus.RECEIVED_BY),
            "order_sent_to": a,
            "order_received_by": a,
        }
    )
    assert (a, "propose_slot", 2) in {(t.party_id, t.intent, t.version) for t in plan(s, g.now).tasks}


def test_broken_pcp_promise_keeps_ladder_then_stalls():
    g = _Log()
    g.add(_ET.ADDRESS_CONFIRMED, {"address": ADDR}).add(_ET.CONSENT_GIVEN, {"scope": "x"})
    g.add(_ET.ORDER_REQUESTED, {"request_ref": "q", "version": 1})
    due = g.now + timedelta(hours=24)
    g.add(_ET.COMMITMENT_MADE, {"cid": "c-v1", "party": "pcp", "what": "fax", "due_at": due})
    for k in (1, 2, 3):
        assert ("pcp", f"followup_order_{k}") in by_intent(plan(g.state, g.now))
        g.add(_ET.COMMITMENT_BROKEN, {"cid": "c-v1"})  # unanswered follow-up
    assert esc_triggers(plan(g.state, g.now)) == [EscalationTrigger.ORDER_STALLED]


def test_signed_swo_keeps_outstanding_addendum_promise_open():
    g = _Log()
    g.add(_ET.ADDRESS_CONFIRMED, {"address": ADDR}).add(_ET.CONSENT_GIVEN, {"scope": "x"})
    g.add(_ET.ORDER_REQUESTED, {"request_ref": "q", "version": 1})
    g.add(_ET.VISIT_NOTE_RECEIVED, {"note": VisitNote(doc_id="n1", text="n", received_at=g.now)})
    g.add(_ET.ADDENDUM_REQUESTED, {"missing": [LCDCriterion.HOME_ACCOMMODATES]})
    g.add(_ET.COMMITMENT_MADE, {"cid": "add-1", "party": "pcp", "what": "addendum", "due_at": g.now})
    g.add(_ET.SWO_SIGNED_RECEIVED, {"swo": full_swo()})
    assert g.state.commitments["add-1"].status == "OPEN"
    g.add(_ET.VISIT_NOTE_RECEIVED, {"note": VisitNote(doc_id="n2", text="a", received_at=g.now)})
    assert g.state.commitments["add-1"].status == "KEPT"
