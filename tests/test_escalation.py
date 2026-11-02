import pytest

from miramace.clock import SIM_START
from miramace.escalation import CATALOGUE, EFFECTS, open_escalation, option_ids, resolve
from miramace.events import EventType
from miramace.models import SWO, CaseState, Escalation, EscalationLane, EscalationTrigger, Track


def with_esc(state: CaseState, ev) -> CaseState:
    esc = Escalation.model_validate(ev.payload["escalation"])
    return state.model_copy(update={"escalations": {**state.escalations, esc.eid: esc}})


def test_catalogue_complete_and_effects_valid():
    assert set(CATALOGUE) == set(EscalationTrigger)
    for (trig, opt), _fn in EFFECTS.items():
        assert opt in option_ids(trig)
    for trig, spec in CATALOGUE.items():
        assert spec.suggested in option_ids(trig)


def test_open_and_resolve_roundtrip():
    s = CaseState(case_id="c1")
    ev = open_escalation(EscalationTrigger.COST_CONCERN, s, SIM_START, {"utterance": "I can't afford that"})
    assert ev.type == EventType.ESCALATION_OPENED and ev.source.kind == "RULE"
    esc = ev.parsed().escalation
    assert esc.eid == "esc-cost_concern-1"
    assert esc.lane == EscalationLane.PATIENT and esc.track == Track.PATIENT
    assert "afford" in esc.question
    # deterministic
    assert (
        open_escalation(EscalationTrigger.COST_CONCERN, s, SIM_START, {"utterance": "I can't afford that"})
        == ev
    )

    s2 = with_esc(s, ev)
    with pytest.raises(ValueError):
        resolve(s2, esc.eid, "pcp_corrected_to_K0001", None, "nurse", SIM_START)
    with pytest.raises(ValueError):
        resolve(s2, "esc-nope-1", "patient_declined", None, "nurse", SIM_START)

    out = resolve(s2, esc.eid, "nurse_resolved_accepted", "talked it through", "nurse", SIM_START)
    assert [e.type for e in out] == [EventType.HUMAN_RESOLVED, EventType.COST_ACCEPTED]
    assert all(e.source.kind == "HUMAN" and e.source.ref == esc.eid for e in out)
    assert out[1].causation_id == out[0].event_id

    out = resolve(s2, esc.eid, "patient_declined", None, "nurse", SIM_START)
    assert out[1].type == EventType.CASE_CANCELLED

    # second escalation of the same trigger gets n=2
    assert (
        open_escalation(EscalationTrigger.COST_CONCERN, s2, SIM_START, {})
        .parsed()
        .escalation.eid.endswith("-2")
    )


def test_code_drift_effect_emits_k0001_swo():
    s = CaseState(case_id="c1", swo=SWO(hcpcs="K0003", signed=True))
    s = with_esc(s, open_escalation(EscalationTrigger.CODE_DRIFT, s, SIM_START, {"hcpcs": "K0003"}))
    out = resolve(s, "esc-code_drift-1", "pcp_corrected_to_K0001", None, "advocate", SIM_START)
    assert out[1].type == EventType.SWO_SIGNED_RECEIVED
    assert out[1].parsed().swo.hcpcs == "K0001"


def test_order_stalled_effect_new_commitment_and_noop_options():
    s = CaseState(case_id="c1")
    s = with_esc(
        s, open_escalation(EscalationTrigger.ORDER_STALLED, s, SIM_START, {"what": "fax", "followups": 3})
    )
    out = resolve(s, "esc-order_stalled-1", "advocate_called_got_owner", "Maria owns it", "adv", SIM_START)
    assert out[1].type == EventType.COMMITMENT_MADE
    assert out[1].parsed().due_at > SIM_START
    s = with_esc(s, open_escalation(EscalationTrigger.SUPPLIERS_EXHAUSTED, s, SIM_START, {}))
    assert len(resolve(s, "esc-suppliers_exhausted-1", "advocate_found_supplier", None, "a", SIM_START)) == 1
