"""B6 — engine units: value parsing, quote re-check, PHI gate, simulators, scenario schema."""

from __future__ import annotations

from datetime import datetime

import pytest

from miramace.clock import SIM_START
from miramace.events import Source, make_event
from miramace.executor import SLOTS, build_call_task, phi_allowed
from miramace.fakes.sim import (
    FactSheetDocReader,
    InboundSimulator,
    SimCallerPort,
    swo_from_claims,
    with_defaults,
)
from miramace.interpret import interpret_call, interpret_doc, parse_int, parse_when
from miramace.models import (
    CHICAGO,
    CallResult,
    Claim,
    InboundDoc,
    InboundDocKind,
    SupplierRecord,
    SupplierStatus,
    Task,
    TaskKind,
    Utterance,
    dr_chen,
    eleanor,
)
from miramace.reducer import initial_state
from miramace.scenario import Scenario, list_scenarios, load_scenario

MON = SIM_START  # Mon 2026-11-02 10:00 CT
LAKE = "lakeshore-home-medical-equipment"


def _dt(*a: int) -> datetime:
    return datetime(*a, tzinfo=CHICAGO)


# ---------------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Thursday", _dt(2026, 11, 5, 17)),
        ("by Friday", _dt(2026, 11, 6, 17)),
        ("tomorrow", _dt(2026, 11, 3, 17)),
        ("tomorrow at 10 AM", _dt(2026, 11, 3, 10)),
        ("in 3 days", _dt(2026, 11, 5, 17)),
        ("within 2 business days", _dt(2026, 11, 4, 17)),
        ("end of day today", _dt(2026, 11, 2, 17)),
        ("Monday", _dt(2026, 11, 9, 17)),  # same weekday -> next week
        ("2:30 pm", _dt(2026, 11, 2, 14, 30)),
        ("2026-11-12", _dt(2026, 11, 12, 17)),
        ("Nov 12", _dt(2026, 11, 12, 17)),
    ],
)
def test_parse_when(text, expected):
    assert parse_when(text, MON) == expected


def test_parse_when_unknown_and_ints():
    assert parse_when("soon-ish", MON) is None
    assert parse_when(None, MON) is None
    assert parse_int("2 days") == 2 and parse_int("three") == 3 and parse_int("none") is None


# ---------------------------------------------------------------------------------- helpers


def _state(**sup: SupplierStatus):
    p = eleanor().model_copy(update={"address": None})
    s = initial_state("c1", p, dr_chen())
    from miramace.models import Address

    s.patient_info.address = Address.parse("2150 N Lincoln Park W, Chicago, IL 60614")
    for sid, st in sup.items():
        s.suppliers[sid.replace("_", "-")] = SupplierRecord(sid=sid.replace("_", "-"), status=st)
    return s


def _task(kind: TaskKind, party: str, intent: str, **args) -> Task:
    return Task(
        task_id="t1",
        case_id="c1",
        kind=kind,
        party_id=party,
        intent=intent,
        idem_key=f"c1:{party}:{intent}:v1",
        due_at=MON,
        args=args,
    )


def _result(lines: list[tuple[str, str]], claims: list[Claim], outcome: str = "COMPLETED") -> CallResult:
    seq: list[tuple[str, str]] = []
    for sp, t in lines:  # every answer follows a caller question (extractor.answer_lines)
        if sp == "counterparty":
            seq.append(("caller", "Could you tell me about that?"))
        seq.append((sp, t))
    tr = [Utterance(turn=i, speaker=sp, name=sp, text=t, ts=i * 6.0) for i, (sp, t) in enumerate(seq)]
    return CallResult(
        task_id="t1",
        outcome=outcome,
        transcript=tr,
        claims=claims,
        flags=[],
        duration_s=60,
        callback_after=None,
        transport="fake",
    )


def _c(field: str, value: str, quote: str) -> Claim:
    return Claim(field=field, status="stated", value=value, evidence_quote=quote)


# ---------------------------------------------------------------------------------- PHI gate


def test_phi_gate():
    s = _state(lakeshore_home_medical_equipment=SupplierStatus.UNCONTACTED)
    screen = _task(TaskKind.CALL_SUPPLIER, LAKE, "screen")
    ct = build_call_task(screen, s)
    assert ct.phi is None and "60614" in ct.slots[2].ask
    assert "Eleanor" not in ct.purpose + " ".join(ct.context.values())
    commit = _task(TaskKind.CALL_SUPPLIER, LAKE, "commit")
    assert not phi_allowed(commit, s)  # not qualified yet
    s.suppliers[LAKE].status = SupplierStatus.QUALIFIED
    assert build_call_task(commit, s).phi is not None
    assert build_call_task(screen, s).phi is None  # screen never carries PHI
    assert build_call_task(_task(TaskKind.CALL_PCP, "pcp", "request_order"), s).phi is not None
    for fam, slots in SLOTS.items():  # asks must be distinct (fake persona matches by ask text)
        asks = [x.ask for x in slots]
        assert len(set(asks)) == len(asks), fam


# ---------------------------------------------------------------------------------- interpret


def test_screen_qualifies_and_fabricated_quote_is_rejected():
    s = _state(lakeshore_home_medical_equipment=SupplierStatus.UNCONTACTED)
    lines = [
        ("counterparty", "Lakeshore Home Medical Equipment, this is Tom."),
        ("counterparty", "Yes, we're taking new Medicare Part B patients."),
        ("counterparty", "Yes, we have standard K0001 chairs in stock."),
        ("counterparty", "Yes, we deliver to 60614."),
        ("counterparty", "Yes, we accept assignment for every rental month."),
    ]
    claims = [
        _c("taking_new_medicare", "yes", "taking new Medicare Part B patients"),
        _c("k0001_in_stock", "yes", "K0001 chairs in stock"),
        _c("delivers_to_zip", "yes", "we deliver to 60614"),
        _c("accepts_assignment_all_months", "yes", "accept assignment for every rental month"),
    ]
    evs = interpret_call(
        _task(TaskKind.CALL_SUPPLIER, LAKE, "screen"), _result(lines, claims), s, MON, transcript_id="tr-1"
    )
    assert [e.type for e in evs] == ["SUPPLIER_QUALIFIED"]
    assert evs[0].source == Source(kind="TRANSCRIPT", ref="tr-1")

    bad = claims[:3] + [_c("accepts_assignment_all_months", "yes", "we accept assignment for all 13 months")]
    evs = interpret_call(
        _task(TaskKind.CALL_SUPPLIER, LAKE, "screen"), _result(lines, bad), s, MON, transcript_id="tr-2"
    )
    types = [e.type for e in evs]
    assert "EXTRACTION_FAILED" in types and "SUPPLIER_QUALIFIED" not in types
    assert "SUPPLIER_NO_ANSWER" in types  # unknown on first screen -> one re-screen


def test_screen_disqualifies_on_explicit_no():
    s = _state(lakeshore_home_medical_equipment=SupplierStatus.UNCONTACTED)
    lines = [("counterparty", "No, we're out of K0001 chairs right now.")]
    claims = [_c("k0001_in_stock", "no", "we're out of K0001 chairs")]
    evs = interpret_call(
        _task(TaskKind.CALL_SUPPLIER, LAKE, "screen"), _result(lines, claims), s, MON, transcript_id="tr"
    )
    assert [e.type for e in evs] == ["SUPPLIER_DISQUALIFIED"]
    assert evs[0].payload["reason"] == "not:k0001_in_stock"


def test_pcp_request_order_parses_promise_in_code():
    s = _state()
    lines = [
        ("counterparty", "We'll fax the visit note and the signed order."),
        ("counterparty", "By Thursday."),
    ]
    claims = [
        _c("promised_action", "fax visit note and signed order", "fax the visit note"),
        _c("promised_by", "Thursday", "By Thursday."),
    ]
    evs = interpret_call(
        _task(TaskKind.CALL_PCP, "pcp", "request_order"), _result(lines, claims), s, MON, transcript_id="tr"
    )
    assert [e.type for e in evs] == ["ORDER_REQUESTED", "COMMITMENT_MADE"]
    assert datetime.fromisoformat(evs[1].payload["due_at"]) == _dt(2026, 11, 5, 17)
    assert evs[1].payload["cid"] == "pcp-req-v1-1"
    # unanswered -> nothing learned
    assert (
        interpret_call(
            _task(TaskKind.CALL_PCP, "pcp", "request_order"),
            _result([], [], "NO_ANSWER"),
            s,
            MON,
            transcript_id="tr",
        )
        == []
    )


def test_doc_interpretation_checks_quotes_against_doc():
    s = _state()
    doc = InboundDoc(
        kind=InboundDocKind.VISIT_NOTE,
        doc_id="d1",
        arrives_at=MON,
        text="She cannot reach the bathroom safely. A walker does not help.",
    )
    claims = [
        Claim(
            field="MRADL_LIMITATION",
            status="stated",
            value="met",
            evidence_quote="She cannot reach the bathroom safely.",
        ),
        Claim(
            field="HOME_ACCOMMODATES",
            status="stated",
            value="met",
            evidence_quote="Her home has wide doorways.",
        ),
    ]  # not in the note
    evs = interpret_doc(doc, claims, s, MON)
    assert [e.type for e in evs] == ["EXTRACTION_FAILED", "VISIT_NOTE_RECEIVED", "LCD_EVALUATED"]
    res = evs[-1].payload["checklist"]["results"]
    assert res["MRADL_LIMITATION"] == "met" and res["HOME_ACCOMMODATES"] == "not_documented"
    assert all(e.source.kind == "INBOUND_DOC" and e.source.ref == "d1" for e in evs)


# ---------------------------------------------------------------------------------- simulators


class _Inner:
    def __init__(self):
        self.n = 0

    async def run_call(self, task):
        self.n += 1
        return _result([("counterparty", "hi")], [])


async def test_sim_caller_no_answer_ghost_and_unknown_party():
    sc = load_scenario("happy_path")
    personas = dict(sc.personas)
    personas[LAKE] = personas[LAKE].model_copy(
        update={
            "sim": personas[LAKE].sim.model_copy(update={"no_answer_attempts": 1, "ghost_after": "commit"})
        }
    )
    inner = _Inner()
    sim = SimCallerPort(personas, inner)
    s = _state(lakeshore_home_medical_equipment=SupplierStatus.UNCONTACTED)
    ct = build_call_task(_task(TaskKind.CALL_SUPPLIER, LAKE, "screen"), s)
    assert (await sim.run_call(ct)).outcome == "NO_ANSWER"
    assert (await sim.run_call(ct)).outcome == "COMPLETED"
    sim.observe(
        make_event(
            "c1",
            "SUPPLIER_COMMITTED",
            {"sid": LAKE, "ref": "r", "expires_at": MON},
            Source(kind="RULE", ref="x"),
            MON,
        )
    )
    assert (await sim.run_call(ct)).outcome == "NO_ANSWER"
    other = ct.model_copy(update={"party_id": "evanston-home-medical"})
    assert (await sim.run_call(other)).outcome == "NO_ANSWER"
    assert inner.n == 1


def test_lost_first_request_and_effective_persona():
    sc = load_scenario("happy_path")
    personas = dict(sc.personas)
    pcp = personas["pcp"]
    personas["pcp"] = pcp.model_copy(update={"sim": pcp.sim.model_copy(update={"lost_first_request": True})})
    inbound, sim = InboundSimulator(personas), SimCallerPort(personas, None)
    req1 = make_event(
        "c1", "ORDER_REQUESTED", {"request_ref": "r", "version": 1}, Source(kind="RULE", ref="x"), MON
    )
    for o in (inbound, sim):
        o.observe(req1)
        o.observe(req1)  # idempotent
    assert inbound.next_arrival("c1") is None
    recv = next(f for f in sim.effective("pcp").facts if f.slot == "request_received")
    assert recv.value == "no"
    req2 = make_event(
        "c1", "ORDER_REQUESTED", {"request_ref": "r", "version": 2}, Source(kind="RULE", ref="y"), MON
    )
    inbound.observe(req2)
    sim.observe(req2)
    assert next(f for f in sim.effective("pcp").facts if f.slot == "request_received").value == "yes"
    assert len(inbound.queue["c1"]) == 2  # note + signed SWO
    note = inbound.queue["c1"][0]
    assert note.kind == InboundDocKind.VISIT_NOTE and note.arrives_at > MON


async def test_doc_reader_and_swo_roundtrip():
    sc = load_scenario("happy_path")
    inbound = InboundSimulator(sc.personas)
    inbound.observe(
        make_event(
            "c1", "ORDER_REQUESTED", {"request_ref": "r", "version": 1}, Source(kind="RULE", ref="x"), MON
        )
    )
    docs = inbound.pop_due("c1", _dt(2026, 12, 1, 9))
    reader = FactSheetDocReader(sc.personas)
    note = next(d for d in docs if d.kind == InboundDocKind.VISIT_NOTE)
    assert len(await reader.read_doc(note)) == 5
    swo_doc = next(d for d in docs if d.kind == InboundDocKind.SIGNED_SWO)
    swo = swo_from_claims(await reader.read_doc(swo_doc))
    assert swo.hcpcs == "K0001" and swo.signed and swo.quantity == 1 and swo.practitioner_npi == "1234567893"


def test_defaults_fill_engine_slots_only():
    spec = load_scenario("ghost_failover").personas[LAKE]
    eff = with_defaults(spec)
    slots = {f.slot for f in eff.facts}
    assert {"reserve_unit", "order_received", "delivery_slot", "delivered", "reference_number"} <= slots
    orig = {f.slot: f.text for f in spec.facts}
    assert all(f.text == orig[f.slot] for f in eff.facts if f.slot in orig)


# ---------------------------------------------------------------------------------- scenario schema


def test_scenarios_load_and_validate():
    names = list_scenarios()
    assert "happy_path" in names
    for n in names:
        sc = load_scenario(n)
        assert sc.name == n and sc.personas["pcp"].party_id == "pcp"
    with pytest.raises(ValueError):
        Scenario.model_validate(
            {
                "name": "x",
                "ground_truth": {"claim_would_pass": True},
                "personas": {LAKE: {"role": "supplier", "org_name": "Wrong Name", "answerer_name": "T"}},
            }
        )


def test_bool_claim_falls_back_to_quote_when_value_not_yes_no():
    """Live-run regression: model returned reserve_unit='K0001 chair' (quote 'Yes, we can hold a K0001 chair')."""
    from miramace.interpret import _bool
    from miramace.models import Claim

    cl = {"reserve_unit": Claim(field="reserve_unit", status="stated", value="K0001 chair",
                                evidence_quote="Yes, we can hold a K0001 chair for her.")}
    assert _bool(cl, "reserve_unit") is True
    cl2 = {"reserve_unit": Claim(field="reserve_unit", status="unknown", value=None, evidence_quote=None)}
    assert _bool(cl2, "reserve_unit") is None
