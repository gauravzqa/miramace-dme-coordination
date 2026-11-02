"""Final-verifier demonstrations. Each test PASSES when the defect it documents is present."""

from __future__ import annotations

from datetime import timedelta

from miramace.calls.runner import guard_caller_turn
from miramace.agents.schemas import CallerTurn
from miramace.clock import SIM_START, VirtualClock
from miramace.events import EventType as ET
from miramace.runloop import run_case
from miramace.scenario import build_ports, load_scenario, run_scenario, seed_case
from miramace.store import Store


async def test_vf_supplier_readback_is_replaced_by_commitment_guard():
    """FIXED: the screening readback echoes 'we accept assignment' but restates the counterparty, so the
    commitment guard leaves it alone; the persona's 'Yes, that's correct' records the readback as affirmed."""
    summary, store, _ = await run_scenario("happy_path", db_dir=None)
    trs = [t for t in store.transcripts(summary.case_id) if t.party_id != "pcp"]
    screen = trs[0]
    texts = [u.text for u in screen.result.transcript if u.speaker == "caller"]
    assert not any("can't agree to anything" in t for t in texts)
    assert any(t.startswith("Just to confirm what I heard") for t in texts)
    assert "commitment_blocked" not in screen.result.flags
    assert "no_readback" not in screen.result.flags
    for t in trs:
        assert "commitment_blocked" not in t.result.flags


def test_vf_guard_blocks_verbatim_readback():
    """FIXED: readback lines pass the guard; first-person commitments on the patient's behalf are still blocked."""
    from miramace.models import CallTask

    task = CallTask(
        task_id="t", party="supplier", party_id="x", org_name="X", phone="", intent="screen",
        purpose="p", slots=[], phi=None, context={},
    )

    def g(phase: str, say: str):
        return guard_caller_turn(
            task, phase, CallerTurn(say=say, slots_touched=[], action="continue", hangup_reason=None)
        )

    rb = "Just to confirm what I heard: Yes, we accept assignment for every rental month. Is that correct?"
    for phase in ("readback", "verified"):
        out, flags = g(phase, rb)
        assert "commitment_blocked" not in flags and out.say == rb
    out, flags = g("closing", "So to confirm: you accept assignment for every month.")
    assert "commitment_blocked" not in flags
    for say in ("She agrees to the rental.", "She'll pay the copay.", "Great, we'll take it.",
                "Go ahead and schedule it for Tuesday.", "Yes, book it.", "Just to confirm, she agrees."):
        out, flags = g("verified", say)
        assert "commitment_blocked" in flags and "can't agree" in out.say, say


def test_vf_voice_pcp_wiring_silently_falls_back_to_text():
    """FIXED (agent L, scenario.build_ports): voice_pcp=True now wires a VoiceCallerPort for PCP calls."""
    sc = load_scenario("happy_path")
    clock = VirtualClock(SIM_START)
    ports = build_ports(sc, Store(":memory:"), clock, voice_pcp=True)
    assert type(ports.caller_pcp.inner).__name__ == "VoiceCallerPort"


async def test_vf_ghost_failover_attempt_counter_repeats():
    """FIXED: unanswered order-receipt checks are numbered 1, 2 from the supplier record; the release reason
    names what failed; no cold backup screens once an order is in flight with a usable supplier (≤ 12 calls)."""
    summary, store, _ = await run_scenario("ghost_failover", db_dir=None)
    ev = store.events(summary.case_id)
    na = [e.payload for e in ev if e.type == ET.SUPPLIER_NO_ANSWER]
    lake = [p["attempt"] for p in na if p["sid"].startswith("lakeshore")]
    assert lake == [1, 2]
    rel = next(e.payload for e in ev if e.type == ET.SUPPLIER_RELEASED)
    assert "order-receipt" in rel["reason"]
    assert summary.status == "DONE" and summary.calls <= 12


async def test_vf_pcp_never_got_it_first_followup_is_six_days_after_promise():
    """FIXED: follow-ups are 8/16/24 business hours; a promise due "by Thursday" (Thu 17:00) is followed up the
    next business morning (Fri 09:00), not ~6 days later."""
    summary, store, _ = await run_scenario("pcp_never_got_it", db_dir=None)
    ev = store.events(summary.case_id)
    promise = next(e for e in ev if e.type == ET.COMMITMENT_MADE)
    verify = next(e for e in ev if e.type == ET.ORDER_REQUEST_VERIFIED)
    due = promise.parsed().due_at
    assert verify.sim_time - due <= timedelta(hours=17)
    assert summary.status == "DONE"


async def test_vf_doc_reader_failure_is_read_as_missing_criteria():
    """FIXED: a doc-reader exception (e.g. OpenAI 401/timeout in live mode) is recorded as EXTRACTION_FAILED,
    retried once, then escalated as EXTRACTION_CHECK_FAILED on the order track — never read as all-missing
    criteria, so the PCP is not asked for an addendum."""
    sc = load_scenario("happy_path")
    store = Store(":memory:")
    clock = VirtualClock(SIM_START)
    ports = build_ports(sc, store, clock)

    class Boom:
        async def read_doc(self, doc):
            raise RuntimeError("401 invalid api key")

    ports.doc_reader = Boom()
    seed_case(store, sc, "vf-doc", clock)
    summary = await run_case("vf-doc", ports, store, clock, max_steps=60)
    ev = store.events("vf-doc")
    types = [e.type for e in ev]
    assert ET.ADDENDUM_REQUESTED not in types and ET.CORRECTION_REQUESTED not in types
    assert ET.LCD_EVALUATED not in types
    fails = [e.payload for e in ev if e.type == ET.EXTRACTION_FAILED and e.payload["field"] == "document"]
    assert len(fails) >= 2 and "401" in fails[0]["reason"]
    escs = [e.parsed().escalation for e in ev if e.type == ET.ESCALATION_OPENED]
    assert any(x.trigger == "EXTRACTION_CHECK_FAILED" and x.track == "order" for x in escs)
    assert summary.status != "DONE"


async def test_vf_companion_promises_nurse_when_intent_rejected(tmp_path):
    """FIXED (mira chat): intents are submitted first; the rejected cost_concern on a DONE case is re-routed as
    needs_human and the companion's "letting a nurse know" reply is replaced by an honest closed-case template."""
    from miramace.cli import _honest_reply, _submit_intents
    from miramace.fakes.companion import KeywordCompanion
    from miramace.intents import IntentAPI
    from miramace.snapshot import build_snapshot

    summary, store, _ = await run_scenario("happy_path", db_dir=None)
    assert summary.status == "DONE"
    clock = VirtualClock(store.events(summary.case_id)[-1].sim_time)
    snap = build_snapshot(store.state(summary.case_id), clock.now())
    reply = await KeywordCompanion(conversation_id="c").respond(snap, [], "I can't afford this")
    acks = _submit_intents(IntentAPI(store, clock), summary.case_id, reply.intents, store)
    assert any(not a.accepted for _, a in acks)
    assert any(a.accepted and i.kind == "needs_human" for i, a in acks)
    say = _honest_reply(reply.reply, acks, store.state(summary.case_id))
    assert say.startswith("Your case is closed") and "let our care team know" in say
