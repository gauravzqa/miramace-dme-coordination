"""B4 — snapshot, grounding, notify, intents, companion, patient/companion fakes."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from miramace.agents.schemas import CompanionOut, IntentLLM
from miramace.clock import SIM_START, VirtualClock
from miramace.companion.agent import LLMCompanion
from miramace.events import Event, EventType
from miramace.fakes.companion import KeywordCompanion
from miramace.fakes.patient import ScriptedPatient
from miramace.grounding import grounding_check
from miramace.intents import IntentAPI, parse_us_address
from miramace.models import (
    CHICAGO,
    Address,
    CaseState,
    Commitment,
    CompanionReply,
    ComposeRequest,
    DeliveryStatus,
    EscalationLane,
    EscalationTrigger,
    Intent,
    OrderStatus,
    Owner,
    PatientStatus,
    SupplierTrackStatus,
    TrackState,
    dr_chen,
    eleanor,
)
from miramace.notify import TEMPLATES, render_template, should_notify
from miramace.ports import CompanionPort, PatientChannel
from miramace.snapshot import build_snapshot, snapshot_text

THU = datetime(2026, 11, 5, 17, 0, tzinfo=CHICAGO)
LAKESHORE = "lakeshore-home-medical-equipment"


def base_state(**kw) -> CaseState:
    p = eleanor()
    p.address = Address.parse("2150 N Lincoln Park W, Chicago, IL 60614")
    st = CaseState(
        case_id="c1",
        patient_info=p,
        pcp=dr_chen(),
        patient=TrackState(state=PatientStatus.CONSENTED, owner=Owner.US),
        order=TrackState(state=OrderStatus.REQUESTED, owner=Owner.PCP, due_at=THU, party_id="pcp"),
        supplier=TrackState(state=SupplierTrackStatus.COMMITTED, owner=Owner.SUPPLIER, party_id=LAKESHORE),
        committed_sid=LAKESHORE,
        commitments={
            "k1": Commitment(cid="k1", party="pcp", what="send note + signed order", due_at=THU),
        },
        patient_flags={"address", "consent"},
    )
    return st.model_copy(update=kw)


# ------------------------------------------------------------------------------------- snapshot


def test_snapshot_labels_and_allowed():
    snap = build_snapshot(base_state(), SIM_START, contacts={"pcp": "Maria"})
    order = snap.items[0]
    assert order.owner_label == "Dr. Chen's office (Maria)"
    assert order.due_label == "by Thursday, Nov 5"
    sup = next(i for i in snap.items if i.track == "supplier")
    assert sup.owner_label == "Lakeshore Home Medical Equipment"
    for d in ("Thursday", "Nov 5", "November 5", "11/5"):
        assert d in snap.allowed_dates
    assert "Maria" in snap.allowed_names and "Dr. Chen" in snap.allowed_names
    assert snap.cost.deductible_usd == 283


def test_snapshot_only_dollar_amount_is_283():
    states = [
        base_state(),
        CaseState(case_id="c0", patient_info=eleanor(), pcp=dr_chen()),
        base_state(
            patient=TrackState(state=PatientStatus.COST_DISCLOSED, owner=Owner.PATIENT),
            delivery=TrackState(state=DeliveryStatus.SCHEDULING, owner=Owner.PATIENT, party_id=LAKESHORE),
            delivery_slot=datetime(2026, 11, 9, 10, 0, tzinfo=CHICAGO),
        ),
    ]
    for st in states:
        snap = build_snapshot(st, SIM_START)
        blob = snapshot_text(snap) + snap.model_dump_json()
        amounts = set(re.findall(r"\$\s?[\d,.]*\d", blob))
        assert amounts <= {"$283"}, amounts
        assert "$283" in snapshot_text(snap)


# ------------------------------------------------------------------------------------ grounding


def test_grounding_flags_invented_amount_and_day():
    snap = build_snapshot(base_state(), SIM_START)
    v = grounding_check("Your chair will cost $150 and arrive Friday.", snap, None)
    assert any("$150" in x for x in v)
    assert any("Friday" in x for x in v)


def test_grounding_passes_snapshot_facts():
    snap = build_snapshot(base_state(), SIM_START, contacts={"pcp": "Maria"})
    text = (
        "Dr. Chen's office (Maria) promised to send the signed order by Thursday, Nov 5. "
        "I'll check with them then. After the $283 deductible you pay 20% for up to 13 months. "
        "Lakeshore Home Medical Equipment will provide your wheelchair, and Medicare covers the rest. "
        "Okay, I'm here if you need me."
    )
    assert grounding_check(text, snap, None) == []
    assert grounding_check(snapshot_text(snap), snap, None) == []


def test_grounding_flags_invented_name_and_percent():
    snap = build_snapshot(base_state(), SIM_START)
    v = grounding_check("I spoke with Jennifer at the office and you'll pay 15%.", snap, None)
    assert any("Jennifer" in x for x in v)
    assert any("15%" in x for x in v)


def test_templates_are_grounded():
    st = base_state(
        delivery=TrackState(state=DeliveryStatus.SCHEDULING, owner=Owner.PATIENT, party_id=LAKESHORE),
        delivery_slot=datetime(2026, 11, 9, 10, 0, tzinfo=CHICAGO),
    )
    snap = build_snapshot(st, SIM_START)
    for purpose in TEMPLATES:
        text = render_template(purpose, snap)
        assert grounding_check(text, snap, None) == [], (purpose, text)
    assert "Monday, Nov 9" in render_template("delivery_slot", snap)
    assert "$283" in render_template("cost_disclosure", snap)


# --------------------------------------------------------------------------------------- notify


def test_notify_cap_and_action_required_bypass():
    noon = SIM_START.replace(hour=12)
    st = base_state(last_notified_at=SIM_START)
    assert should_notify(st, "status_update", noon) is False
    assert should_notify(st, "cost_disclosure", noon) is True
    assert should_notify(st, "delivery_slot", noon) is True
    assert should_notify(st, "status_update", noon + timedelta(days=1)) is True
    assert should_notify(base_state(), "status_update", noon) is True
    # quiet hours apply to everything
    late = SIM_START.replace(hour=22)
    assert should_notify(base_state(), "welcome_intake", late) is False


# --------------------------------------------------------------------------------------- intents


class FakeStore:
    def __init__(self, state: CaseState):
        self._state = state
        self.events: list[Event] = []

    def state(self, case_id: str) -> CaseState:
        return self._state

    def append(self, event: Event) -> Event:
        self.events.append(event)
        return event


def _intent(kind: str, **kw) -> Intent:
    return Intent(kind=kind, utterance=kw.pop("utterance", "..."), conversation_id="conv-1", **kw)


def test_cost_concern_opens_patient_lane_escalation():
    store = FakeStore(base_state(patient=TrackState(state=PatientStatus.COST_DISCLOSED, owner=Owner.PATIENT)))
    api = IntentAPI(store, VirtualClock())
    ack = api.submit("c1", _intent("cost_concern", utterance="I can't afford that"))
    assert ack.accepted
    assert ack.events == ["CONCERN_RAISED", "ESCALATION_OPENED"]
    esc_ev = store.events[1]
    esc = esc_ev.parsed().escalation
    assert esc.lane == EscalationLane.PATIENT and esc.trigger == EscalationTrigger.COST_CONCERN
    assert esc_ev.source.kind == "PATIENT_INTENT" and esc_ev.source.ref == "conv-1"
    assert store.events[0].parsed().kind == "cost"


def test_cost_accepted_rejected_before_disclosure():
    store = FakeStore(base_state())
    ack = IntentAPI(store, VirtualClock()).submit("c1", _intent("cost_accepted"))
    assert not ack.accepted and ack.reason and store.events == []
    store2 = FakeStore(
        base_state(patient=TrackState(state=PatientStatus.COST_DISCLOSED, owner=Owner.PATIENT))
    )
    ack2 = IntentAPI(store2, VirtualClock()).submit("c1", _intent("cost_accepted"))
    assert ack2.accepted and ack2.events == ["COST_ACCEPTED"]


def test_address_intent_requires_zip():
    store = FakeStore(CaseState(case_id="c1", patient_info=eleanor(), pcp=dr_chen()))
    api = IntentAPI(store, VirtualClock())
    bad = api.submit("c1", _intent("confirm_address", address="2150 N Lincoln Park W, Chicago"))
    assert not bad.accepted and "ZIP" in (bad.reason or "")
    good = api.submit("c1", _intent("confirm_address", address="2150 N Lincoln Park W, Chicago, IL 60614"))
    assert good.accepted and good.events == ["ADDRESS_CONFIRMED"]
    assert store.events[0].parsed().address.zip == "60614"
    assert parse_us_address("12 Oak St, Evanston IL 60201").zip == "60201"


def test_slot_and_delivery_intents_validated():
    store = FakeStore(base_state())
    api = IntentAPI(store, VirtualClock())
    assert not api.submit("c1", _intent("confirm_delivery_slot")).accepted
    assert not api.submit("c1", _intent("delivery_confirmed")).accepted
    assert not api.submit("c1", _intent("reschedule")).accepted
    slot = datetime(2026, 11, 9, 10, 0, tzinfo=CHICAGO)
    store2 = FakeStore(
        base_state(
            delivery=TrackState(state=DeliveryStatus.SCHEDULING, owner=Owner.PATIENT, party_id=LAKESHORE),
            delivery_slot=slot,
        )
    )
    ack = IntentAPI(store2, VirtualClock()).submit("c1", _intent("confirm_delivery_slot"))
    assert ack.events == ["DELIVERY_SCHEDULED"]
    assert store2.events[0].parsed().slot == slot


def test_submit_never_raises_on_store_rejection():
    class Rejecting(FakeStore):
        def append(self, event):
            raise ValueError("IllegalTransition")

    ack = IntentAPI(Rejecting(base_state()), VirtualClock()).submit("c1", _intent("ask_status"))
    assert not ack.accepted and "IllegalTransition" in ack.reason


# ------------------------------------------------------------------------------------- companion


class StubLLM:
    """Tiny scripted LLMPort: returns queued CompanionOut objects in order, records calls."""

    def __init__(self, *outs: CompanionOut):
        self.outs = list(outs)
        self.calls: list[dict] = []

    async def structured(self, *, agent, key, system, messages, schema):
        self.calls.append({"agent": agent, "key": key, "system": system, "messages": messages})
        assert schema is CompanionOut
        return self.outs.pop(0)


def test_stub_and_fakes_satisfy_ports():
    assert isinstance(LLMCompanion(StubLLM()), CompanionPort)
    assert isinstance(KeywordCompanion(), CompanionPort)
    assert isinstance(ScriptedPatient({}), PatientChannel)


async def test_safety_net_forces_needs_human():
    snap = build_snapshot(base_state(), SIM_START)
    llm = StubLLM(CompanionOut(reply="I understand. Your order is being prepared.", intents=[]))
    r = await LLMCompanion(llm).respond(snap, [], "I'm scared I'll fall")
    assert [i.kind for i in r.intents] == ["needs_human"]
    assert "safety_net_needs_human" in r.flags and "nurse" in r.reply.lower()
    assert r.intents[0].utterance == "I'm scared I'll fall"
    assert "CASE SNAPSHOT" in llm.calls[0]["system"]


async def test_regenerate_then_pass():
    snap = build_snapshot(base_state(), SIM_START)
    llm = StubLLM(
        CompanionOut(reply="It will arrive Friday.", intents=[]),
        CompanionOut(
            reply="Dr. Chen's office expects to send it by Thursday, Nov 5.",
            intents=[IntentLLM(kind="ask_status", address=None, preferred_time=None, note=None)],
        ),
    )
    r = await LLMCompanion(llm).respond(snap, [], "Where's my wheelchair?")
    assert r.grounding_ok and "Thursday" in r.reply and "grounding_regenerated" in r.flags
    assert "grounding_hold" not in r.flags
    assert "Friday" in llm.calls[1]["messages"][-1]["content"]
    assert [i.kind for i in r.intents] == ["ask_status"]


async def test_regenerate_then_hold():
    snap = build_snapshot(base_state(), SIM_START)
    llm = StubLLM(
        CompanionOut(reply="It costs $150.", intents=[]),
        CompanionOut(reply="It arrives Friday for $150.", intents=[]),
    )
    r: CompanionReply = await LLMCompanion(llm).respond(snap, [], "How much and when?")
    assert "grounding_hold" in r.flags
    assert "$150" not in r.reply and "Friday" not in r.reply
    assert r.reply.startswith(snap.headline)
    assert r.grounding_ok
    assert [c["key"] for c in llm.calls] == ["respond", "respond_retry"]


async def test_compose_falls_back_to_template():
    snap = build_snapshot(base_state(), SIM_START)
    req = ComposeRequest(case_id="c1", purpose="cost_disclosure", snapshot=snap)
    llm = StubLLM(
        CompanionOut(reply="You'll pay $99.", intents=[]), CompanionOut(reply="$99 total.", intents=[])
    )
    text = await LLMCompanion(llm).compose(req)
    assert text == render_template("cost_disclosure", snap)
    good = StubLLM(CompanionOut(reply="After the $283 deductible you pay 20%. Does that work?", intents=[]))
    assert "$283" in await LLMCompanion(good).compose(req)


# ----------------------------------------------------------------------------------------- fakes


async def test_keyword_companion():
    snap = build_snapshot(base_state(), SIM_START)
    kc = KeywordCompanion()
    r = await kc.respond(snap, [], "That's too expensive, I can't afford it")
    assert [i.kind for i in r.intents] == ["cost_concern"]
    r = await kc.respond(snap, [], "Can I talk to a real person?")
    assert [i.kind for i in r.intents] == ["needs_human"]
    r = await kc.respond(snap, [], "Where's my wheelchair?")
    assert [i.kind for i in r.intents] == ["ask_status"]
    assert "Thursday, Nov 5" in r.reply and grounding_check(r.reply, snap, None) == []
    await kc.compose(ComposeRequest(case_id="c1", purpose="cost_disclosure", snapshot=snap))
    r = await kc.respond(snap, [], "Yes, okay")
    assert [i.kind for i in r.intents] == ["cost_accepted"]
    r = await kc.respond(snap, [], "I live at 2150 N Lincoln Park W, Chicago, IL 60614")
    assert r.intents[0].kind == "confirm_address" and r.intents[0].address.endswith("60614")


async def test_scripted_patient():
    sp = ScriptedPatient(
        {
            "welcome_intake": [
                {"kind": "confirm_address", "address": "2150 N Lincoln Park W, Chicago, IL 60614"},
                {"kind": "give_consent"},
            ]
        },
        conversation_id="p1",
    )
    r = await sp.deliver("c1", "welcome_intake", "Hi", SIM_START)
    assert r is not None and [i.kind for i in r.intents] == ["confirm_address", "give_consent"]
    assert all(i.conversation_id == r.conversation_id for i in r.intents)
    assert await sp.deliver("c1", "cost_disclosure", "cost", SIM_START) is None
    assert len(sp.inbox) == 2


def test_event_types_used_exist():
    for et in (
        "CONCERN_RAISED",
        "ESCALATION_OPENED",
        "DELIVERY_SCHEDULED",
        "POD_CONFIRMED",
        "DELIVERY_MISSED",
    ):
        EventType(et)


def test_local_escalation_fallback_matches_catalogue_ids():
    from miramace.intents import _local_escalation

    st = base_state()
    for trig in (EscalationTrigger.COST_CONCERN, EscalationTrigger.NEEDS_HUMAN):
        esc = _local_escalation(st, trig, {"utterance": "help"})
        assert esc.lane == EscalationLane.PATIENT and esc.suggested in {o.option_id for o in esc.options}
        try:
            from miramace.escalation import option_ids
        except ImportError:
            continue
        assert [o.option_id for o in esc.options] == option_ids(trig)
