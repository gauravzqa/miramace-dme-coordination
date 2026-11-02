"""B3 — runner, phases, brains, TextCallerPort (all offline)."""

from __future__ import annotations

from datetime import datetime

from _b3_data import ORG, maria, pcp_task
from miramace.agents.prompts import DISCLOSURE, INJECTION_TEXT, caller_system, contains_phi
from miramace.agents.schemas import CallerTurn, PersonaTurn
from miramace.calls.brains import FakeCallerBrain, FakePersonaBrain, LLMCallerBrain, LLMPersonaBrain
from miramace.calls.runner import (
    RawCall,
    TextCallerPort,
    Transcript,
    finalize_call,
    guard_caller_turn,
    phase_from,
    run_text_call,
)
from miramace.extract.extractor import FakeExtractor
from miramace.llm.port import ScriptedLLM
from miramace.models import CHICAGO, eleanor
from miramace.ports import Brain, CallerPort


def _callers(raw):
    return [u for u in raw.utterances if u.speaker == "caller"]


async def test_helpful_maria_completes_with_disclosure_first():
    task = pcp_task()
    raw = await run_text_call(task, FakeCallerBrain(), FakePersonaBrain(maria()))
    callers = _callers(raw)
    assert raw.utterances[0].speaker == "counterparty"
    assert raw.utterances[0].text == f"{ORG}, this is Maria."
    assert callers[0].text == DISCLOSURE
    assert raw.end_reason == "complete"
    assert len(callers) <= 14
    assert raw.duration_s == 6 * len(raw.utterances)
    res = await finalize_call(task, raw, FakeExtractor({"pcp": maria()}))
    assert res.outcome == "COMPLETED"
    stated = {c.field: c.value for c in res.claims if c.status == "stated"}
    assert stated["request_received"] == "yes"
    assert stated["order_status"] == "pending signature"
    assert stated["promised_by"] == "Thursday"
    assert not [f for f in res.flags if f.startswith(("quote_unverified", "two_pass"))]
    # PHI only after verification
    pre = [u for u in callers if contains_phi(u.text, task.phi)]
    assert pre and all(u.turn > 0 for u in pre)


async def test_transfer_reemits_disclosure():
    task = pcp_task()
    raw = await run_text_call(task, FakeCallerBrain(), FakePersonaBrain(maria("transfer")))
    texts = [u.text for u in _callers(raw)]
    assert texts.count(DISCLOSURE) == 2
    assert any(u.event == "transfer" for u in raw.utterances)
    assert any(u.name == "Jordan" for u in raw.utterances)
    res = await finalize_call(task, raw, FakeExtractor({"pcp": maria()}))
    assert res.outcome == "COMPLETED"
    # the question asked right before the transfer was re-asked and answered
    assert {c.field for c in res.claims if c.status == "stated"} >= {"request_received", "order_status"}


async def test_ivr_transfers_to_front_desk():
    task = pcp_task()
    raw = await run_text_call(task, FakeCallerBrain(), FakePersonaBrain(maria("ivr")))
    assert any(u.event == "transfer" for u in raw.utterances)
    assert any(u.speaker == "caller" and u.text == "Front desk." for u in raw.utterances)
    assert [u.text for u in _callers(raw)].count(DISCLOSURE) == 2
    # no PHI said to the phone menu
    first_transfer = next(i for i, u in enumerate(raw.utterances) if u.event == "transfer")
    assert not any(contains_phi(u.text, task.phi) for u in raw.utterances[:first_transfer])
    assert raw.end_reason == "complete"


async def test_callback_later_sets_outcome_and_time():
    task = pcp_task()
    port = TextCallerPort(
        FakeCallerBrain,
        lambda pid: FakePersonaBrain(maria("callback_later")),
        FakeExtractor(),
        clock=type("C", (), {"now": staticmethod(lambda: datetime(2026, 11, 2, 10, 0, tzinfo=CHICAGO))})(),
    )
    res = await port.run_call(task)
    assert res.outcome == "CALLBACK_REQUESTED"
    assert res.callback_after == datetime(2026, 11, 2, 14, 0, tzinfo=CHICAGO)


async def test_injection_never_evidence_and_no_phi_leak():
    task = pcp_task()
    raw = await run_text_call(task, FakeCallerBrain(), FakePersonaBrain(maria("injection")))
    inj = next(i for i, u in enumerate(raw.utterances) if u.text == INJECTION_TEXT)
    assert "injection_suspected" in raw.flags
    # caller never reads the Medicare number after the injection
    assert not any(task.phi.mbi in u.text for u in raw.utterances[inj:] if u.speaker == "caller")
    for ex in (FakeExtractor(), FakeExtractor({"pcp": maria()})):
        res = await finalize_call(task, raw, ex)
        for c in res.claims:
            if c.status == "stated":
                assert c.evidence_quote != INJECTION_TEXT
                assert "system note" not in (c.evidence_quote or "").lower()
        os_claim = next(c for c in res.claims if c.field == "order_status")
        assert (os_claim.value or "").lower() != "signed"


async def test_contradicted_readback_downgrades():
    task = pcp_task()
    raw = await run_text_call(task, FakeCallerBrain(), FakePersonaBrain(maria(contradicts_on_readback=True)))
    res = await finalize_call(task, raw, FakeExtractor({"pcp": maria()}))
    assert "readback_not_affirmed" in res.flags
    assert all(c.status == "unknown" for c in res.claims)


def test_phase_progression_and_verification_reset():
    task = pcp_task()
    t = Transcript()
    assert phase_from(t, task) == "opening"
    t.add("counterparty", "IVR", "Thank you for calling. Press 1.")
    t.add("caller", "ai", DISCLOSURE)
    assert phase_from(t, task) == "verify_party"
    t.add("counterparty", "Maria", f"{ORG.upper()}!! this is Maria")
    assert phase_from(t, task) == "verified"
    for s in task.slots:
        if s.required:
            t.add("caller", "ai", s.ask, slots_touched=[s.id])
            t.add("counterparty", "Maria", "ok")
    assert phase_from(t, task) == "readback"
    t.add("caller", "ai", "Just to confirm ... Is that correct?")
    t.add("counterparty", "Maria", "Yes")
    assert phase_from(t, task) == "closing"
    t.add("counterparty", "Maria", "transferring", event="transfer")
    assert phase_from(t, task) == "verify_party"


def test_caller_prompt_phi_gate():
    task = pcp_task()
    phi = task.phi
    for phase in ("opening", "verify_party"):
        s = caller_system(task, phase, task.slots)
        assert not contains_phi(s, phi)
        for needle in (phi.name, phi.mbi, "1954"):
            assert needle not in s
    for phase in ("verified", "readback", "closing"):
        s = caller_system(task, phase, task.slots)
        assert phi.name in s and phi.mbi in s and "1954" in s
    # no PHI in task -> no block even when verified
    assert eleanor().name not in caller_system(pcp_task(phi=False), "verified", [])


def test_guard_blocks_phi_and_commitments():
    task = pcp_task()
    bad = CallerTurn(
        say=f"Calling about {task.phi.name}.", slots_touched=["x"], action="continue", hangup_reason=None
    )
    out, flags = guard_caller_turn(task, "verify_party", bad)
    assert "phi_blocked" in flags and not contains_phi(out.say, task.phi)
    ok, flags = guard_caller_turn(task, "verified", bad)
    assert flags == [] and ok.slots_touched == []
    c = CallerTurn(say="Yes, she agrees to pay.", slots_touched=[], action="continue", hangup_reason=None)
    out, flags = guard_caller_turn(task, "verified", c)
    assert "commitment_blocked" in flags


async def test_llm_brains_with_scripted_llm_and_phi_prompts():
    task = pcp_task()
    spec = maria()
    fake_p = FakePersonaBrain(spec)
    fake_c = FakeCallerBrain()

    # The scripted LLM's answers are produced on the fly by the rule-based brains (same view).
    llm = ScriptedLLM()

    class Probe:
        def __init__(self, inner, real, kind):
            self.inner, self.real, self.kind = inner, real, kind
            self.spec = getattr(real, "spec", None)

        async def next(self, view):
            llm.scripts[(self.kind, f"{task.task_id}:t{len(view.utterances)}")] = await self.inner.next(view)
            return await self.real.next(view)

    caller = LLMCallerBrain(llm)
    persona = LLMPersonaBrain(llm, spec)
    pc, pp = Probe(fake_c, caller, "caller"), Probe(fake_p, persona, "persona")
    assert isinstance(caller, Brain) and isinstance(persona, Brain)
    raw = await run_text_call(task, pc, pp)
    assert raw.end_reason == "complete"
    caller_reqs = [r for r in llm.requests if r["agent"] == "caller"]
    assert caller_reqs
    phi = task.phi
    for r in caller_reqs:
        phase = r["system"].split("PHASE: ")[1].split(".")[0]
        if phase in ("opening", "verify_party"):
            assert not contains_phi(r["system"], phi)
        else:
            assert phi.mbi in r["system"]
    assert all(c.ok for c in llm.ledger)
    keys = [c.key for c in llm.ledger]
    assert len(keys) == len(set((c.agent, c.key) for c in llm.ledger))


async def test_ivr_pre_verify_prompt_has_no_phi_via_llm_caller():
    task = pcp_task()
    llm = ScriptedLLM(
        {
            ("caller", "*"): CallerTurn(
                say="Front desk.", slots_touched=[], action="continue", hangup_reason=None
            )
        }
    )
    brain = LLMCallerBrain(llm)
    t = Transcript()
    t.add("counterparty", "IVR", "Thank you for calling. Press 1.")
    t.add("caller", "ai", DISCLOSURE)
    t.add("counterparty", "IVR", "Say front desk.")
    await brain.next(t.view(task, phase_from(t, task), "caller"))
    assert phase_from(t, task) == "verify_party"
    assert not contains_phi(llm.requests[-1]["system"], task.phi)


async def test_text_caller_port_is_caller_port_and_collects_usage():
    from miramace.agents.schemas import ClaimLLM, Extraction
    from miramace.extract.extractor import Extractor

    task = pcp_task()
    llm = ScriptedLLM(
        {
            ("extractor", f"{task.task_id}:passA"): Extraction(
                claims=[
                    ClaimLLM(
                        field="promised_by", status="stated", value="Thursday", evidence_quote="By Thursday."
                    )
                ]
            ),
            ("extractor", f"{task.task_id}:passB"): Extraction(
                claims=[
                    ClaimLLM(
                        field="promised_by", status="stated", value="thursday", evidence_quote="by thursday"
                    )
                ]
            ),
        }
    )
    port = TextCallerPort(lambda: FakeCallerBrain(), lambda pid: FakePersonaBrain(maria()), Extractor(llm))
    assert isinstance(port, CallerPort)
    res = await port.run_call(task)
    assert res.outcome in ("COMPLETED", "FAILED")
    assert [c.agent for c in res.usage] == ["extractor", "extractor"]
    pb = next(c for c in res.claims if c.field == "promised_by")
    assert pb.status == "stated"


async def test_persona_hangup_without_info_fails():
    task = pcp_task()

    class Hanger:
        spec = maria()

        async def next(self, view):
            if not view.utterances:
                return PersonaTurn(say=f"{ORG}.", event="none", transfer_to=None, revealed_fact_ids=[])
            return PersonaTurn(say="Not interested.", event="hangup", transfer_to=None, revealed_fact_ids=[])

    raw = await run_text_call(task, FakeCallerBrain(), Hanger())
    assert isinstance(raw, RawCall) and raw.end_reason == "counterparty_hangup"
    res = await finalize_call(task, raw, FakeExtractor())
    assert res.outcome == "FAILED" and "counterparty_hangup" in res.flags


async def test_max_turns_stops():
    task = pcp_task(max_turns=3)

    class Chatty:
        async def next(self, view):
            return CallerTurn(
                say="Hmm, interesting?", slots_touched=[], action="continue", hangup_reason=None
            )

    raw = await run_text_call(task, Chatty(), FakePersonaBrain(maria()))
    assert raw.end_reason == "max_turns" and "max_turns" in raw.flags
    assert len(_callers(raw)) == 3
