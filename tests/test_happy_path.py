"""B6 — end-to-end happy path on fakes: DONE, 0 escalations, invariants clean, traceable, PHI gate,
determinism, resume after a human resolution."""

from __future__ import annotations

import time

from miramace.clock import SIM_START, VirtualClock
from miramace.metrics import aggregate, score
from miramace.models import CallTask
from miramace.runloop import resume_after_resolution, run_case
from miramace.scenario import build_ports, load_scenario, open_case, run_scenario, seed_case
from miramace.store import Store


async def test_happy_path_done_fast_and_clean():
    t0 = time.perf_counter()
    summary, store, sc = await run_scenario("happy_path", db_dir=None)
    assert time.perf_counter() - t0 < 3.0
    assert summary.status == "DONE", summary
    assert summary.escalations == [] and summary.open_escalations == []
    assert summary.invariant_violations == []
    assert summary.calls >= 6
    s = score(summary, sc, store.state(summary.case_id), store)
    assert s.passed and s.autonomous_close and not s.false_close and s.phi_before_qualification == 0
    m = aggregate([s])
    assert m.all_passed and m.escalation_precision == 1.0 and m.escalation_recall == 1.0


async def test_every_transcript_event_is_traceable():
    summary, store, _ = await run_scenario("happy_path", db_dir=None)
    tids = store.transcript_ids(summary.case_id)
    assert len(tids) > 0
    events = store.events(summary.case_id)
    tr_events = [e for e in events if e.source.kind == "TRANSCRIPT"]
    assert tr_events and all(e.source.ref in tids for e in tr_events)
    for e in events:
        if e.type == "CALL_COMPLETED":
            assert e.payload["transcript_id"] in tids
    kinds = {e.source.kind for e in events}
    assert {"TRANSCRIPT", "INBOUND_DOC", "PATIENT_INTENT", "RULE"} <= kinds


async def test_screening_calls_carry_no_phi_and_later_calls_do():
    sc = load_scenario("happy_path")
    store, clock = Store(":memory:"), VirtualClock(SIM_START)
    ports = build_ports(sc, store, clock)
    seen: list[CallTask] = []
    inner = ports.caller_supplier.run_call

    async def spy(task: CallTask):
        seen.append(task)
        return await inner(task)

    ports.caller_supplier.run_call = spy  # type: ignore[method-assign]
    seed_case(store, sc, "hp-phi", clock)
    summary = await run_case("hp-phi", ports, store, clock)
    assert summary.status == "DONE"
    screens = [t for t in seen if t.intent == "screen"]
    assert screens and all(t.phi is None for t in screens)
    assert any(t.phi is not None for t in seen if t.intent == "commit")


async def test_fake_runs_are_deterministic():
    a, sa, _ = await run_scenario("happy_path", db_dir=None)
    b, sb, _ = await run_scenario("happy_path", db_dir=None)
    ta = [(e.type, e.sim_time) for e in sa.events(a.case_id)]
    tb = [(e.type, e.sim_time) for e in sb.events(b.case_id)]
    assert ta == tb


async def test_resume_after_cost_concern_resolution(tmp_path):
    sc = load_scenario("happy_path")
    sc = sc.model_copy(
        update={"patient_script": {**sc.patient_script, "cost_disclosure": [{"kind": "cost_concern"}]}}
    )
    store, clock = Store(tmp_path / "cc-1.db"), VirtualClock(SIM_START)
    ports = build_ports(sc, store, clock)
    seed_case(store, sc, "cc-1", clock)
    first = await run_case("cc-1", ports, store, clock)
    assert first.status == "BLOCKED_ON_HUMAN"
    assert first.escalations == ["COST_CONCERN"] and first.invariant_violations == []
    eid = first.open_escalations[0]
    store.close()

    # re-open the DB like the CLI would, with fresh simulators (they are primed from the log)
    store2, clock2 = open_case("cc-1", db_dir=tmp_path)
    ports2 = build_ports(sc, store2, clock2)
    done = await resume_after_resolution(
        "cc-1", eid, "nurse_resolved_accepted", "spoke with her", "nurse", ports2, store2, clock2
    )
    assert done.status == "DONE", done
    assert done.open_escalations == [] and done.invariant_violations == []
    types = [e.type for e in store2.events("cc-1")]
    assert types.count("PATIENT_NOTIFIED") >= 3 and "HUMAN_RESOLVED" in types and types[-1] == "CASE_DONE"
    # docs that arrived before the restart were not delivered twice
    assert types.count("SWO_SIGNED_RECEIVED") == 1 and types.count("VISIT_NOTE_RECEIVED") == 1
