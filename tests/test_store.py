"""B1 — Store: event-sourcing round trip, seq assignment, rejection, task reconcile, transcripts, caches."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from miramace.clock import SIM_START
from miramace.events import Event, EventType, Source, make_event
from miramace.models import (
    CallResult,
    Claim,
    EscalationTrigger,
    TaskKind,
    TaskSpec,
    TaskStatus,
    Track,
    Utterance,
    idem_key,
    load_suppliers,
)
from miramace.reducer import IllegalTransition, reduce_all
from miramace.store import Store
from test_reducer import ADDR, CASE, Log, esc, happy_path, opened

ET = EventType


def spec(intent: str, version: int = 1, due_h: int = 1, party: str = "pcp") -> TaskSpec:
    return TaskSpec(
        kind=TaskKind.CALL_PCP,
        party_id=party,
        intent=intent,
        version=version,
        idem_key=idem_key(CASE, party, intent, version),
        due_at=SIM_START + timedelta(hours=due_h),
        args={"x": 1},
    )


def test_round_trip_file_db(tmp_path):
    db = tmp_path / "runs" / "case.db"
    log = happy_path()
    with Store(db) as s:
        s.create_case(CASE, "happy_path", {"note": "x"}, SIM_START)
        stored = s.append_many(log.events)
        assert [e.seq for e in stored] == list(range(1, len(log.events) + 1))
        mode = s._db.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        expected = s.state(CASE)
    with Store(db) as s2:  # fresh process view: rebuilt from events only
        evs = s2.events(CASE)
        assert [e.event_id for e in evs] == [e.event_id for e in log.events]
        assert evs[3].payload == log.events[3].payload and evs[3].sim_time == log.events[3].sim_time
        assert s2.state(CASE) == expected == reduce_all(evs)
        assert s2.case(CASE)["scenario"] == "happy_path"
        assert s2.case_ids() == [CASE]


def test_append_assigns_seq_and_keeps_source_span():
    s = Store()
    log = opened()
    e1 = s.append(log.events[0])
    e = make_event(
        CASE, ET.CONSENT_GIVEN, {"scope": "x"}, Source(kind="TRANSCRIPT", ref="tr-1", span=(2, 4)), log.t
    )
    e2 = s.append(e)
    assert (e1.seq, e2.seq) == (1, 2) and e.seq == 0  # input not mutated
    got = s.events(CASE)[1]
    assert got.source.span == (2, 4) and got.source.ref == "tr-1"
    assert s.state(CASE).last_seq == 2


def test_first_event_must_be_case_opened():
    s = Store()
    with pytest.raises(IllegalTransition):
        s.append(Log().ev(ET.CONSENT_GIVEN, {"scope": "x"}))
    assert s.events(CASE) == []


def test_rejection_leaves_db_unchanged():
    s = Store()
    log = opened().add(ET.ADDRESS_CONFIRMED, {"address": ADDR})
    s.append_many(log.events)
    before_events, before_state = s.events(CASE), s.state(CASE)
    bad = Log(t=log.t).ev(ET.CASE_DONE)  # CASE_DONE before POD_CONFIRMED
    with pytest.raises(IllegalTransition):
        s.append(bad)
    # batch with a legal then an illegal event: all or nothing
    good = Log(t=log.t).ev(ET.CONSENT_GIVEN, {"scope": "x"})
    with pytest.raises(IllegalTransition):
        s.append_many([good, bad])
    assert s.events(CASE) == before_events and s.state(CASE) == before_state
    # bad payload is rejected too (validated via PAYLOADS)
    tampered = good.model_copy(update={"payload": {"nope": 1}})
    with pytest.raises(ValidationError):
        s.append(tampered)
    # duplicate event_id rejected
    with pytest.raises(ValueError):
        s.append(log.events[1])
    assert s.events(CASE) == before_events
    s.append(good)
    assert s.events(CASE)[-1].seq == 3


def test_state_returns_copy():
    s = Store()
    s.append_many(opened().events)
    st = s.state(CASE)
    st.case.state = "HACKED"
    assert s.state(CASE).case.state == "ACTIVE"
    with pytest.raises(KeyError):
        s.state("nope")


def test_no_event_update_or_delete_api():
    names = [n for n in dir(Store) if not n.startswith("_")]
    assert not [n for n in names if "delete" in n.lower() or n.lower().startswith("update")]


def test_reconcile_idempotent_cancel_and_done_untouched():
    s = Store()
    s.append_many(opened().events)
    specs = [spec("request_order"), spec("screen", party="alpha", due_h=2)]
    p1 = s.reconcile_tasks(CASE, specs, SIM_START)
    p2 = s.reconcile_tasks(CASE, specs, SIM_START)
    assert [t.idem_key for t in p1] == [t.idem_key for t in p2] and len(p2) == 2
    assert len(s.tasks(CASE)) == 2  # no duplicates
    assert p1[0].args == {"x": 1} and p1[0].due_at == specs[0].due_at
    # planner recomputing a later due_at for the same key does not push the task out
    later = [specs[0].model_copy(update={"due_at": SIM_START + timedelta(hours=9)}), specs[1]]
    assert s.reconcile_tasks(CASE, later, SIM_START)[0].due_at == specs[0].due_at
    # DONE untouched, dropped spec -> CANCELLED
    t_done = s.next_pending_due(CASE)
    assert t_done.intent == "request_order"
    s.mark_task(t_done.task_id, TaskStatus.DONE, attempts=1, last_result="COMPLETED")
    pend = s.reconcile_tasks(CASE, [], SIM_START)
    assert pend == []
    by_key = {t.idem_key: t for t in s.tasks(CASE)}
    assert by_key[specs[0].idem_key].status == TaskStatus.DONE and by_key[specs[0].idem_key].attempts == 1
    assert by_key[specs[1].idem_key].status == TaskStatus.CANCELLED
    assert s.task_last_result(t_done.task_id) == "COMPLETED"
    # re-desired DONE key is not re-run; re-desired CANCELLED key is revived
    pend = s.reconcile_tasks(CASE, specs, SIM_START)
    assert [t.idem_key for t in pend] == [specs[1].idem_key]
    assert len(s.tasks(CASE)) == 2


def test_next_pending_due_orders_by_time_and_mark_task():
    s = Store()
    s.reconcile_tasks(CASE, [spec("a", due_h=5), spec("b", due_h=1), spec("c", due_h=3)], SIM_START)
    assert s.next_pending_due(CASE).intent == "b"
    assert [t.intent for t in s.tasks(CASE, TaskStatus.PENDING)] == ["b", "c", "a"]
    s.mark_task(s.next_pending_due(CASE).task_id, TaskStatus.FAILED)
    assert s.next_pending_due(CASE).intent == "c"
    assert s.next_pending_due("other") is None
    with pytest.raises(KeyError):
        s.mark_task("missing", TaskStatus.DONE)


def test_transcripts_round_trip():
    s = Store()
    res = CallResult(
        task_id="t1",
        outcome="COMPLETED",
        transcript=[Utterance(turn=0, speaker="caller", name="Mira", text="Hi", ts=0.0)],
        claims=[Claim(field="k0001_in_stock", status="stated", value="yes", evidence_quote="yes we have it")],
        flags=["readback_ok"],
        duration_s=120,
        callback_after=None,
        transport="fake",
    )
    tid = s.save_transcript(CASE, "t1", "alpha", res, SIM_START)
    rec = s.transcript(tid)
    assert rec.result == res and rec.party_id == "alpha" and rec.sim_time == SIM_START
    assert [r.transcript_id for r in s.transcripts(CASE)] == [tid]
    assert s.transcript_ids(CASE) == {tid} and s.transcript("nope") is None


def test_suppliers_and_projection_caches():
    s = Store()
    sups = load_suppliers()
    s.upsert_suppliers(sups)
    s.upsert_suppliers(sups[:1])
    assert len(s.suppliers()) == len(sups)
    log = opened()
    log.add(ET.COMMITMENT_MADE, {"cid": "c1", "party": "pcp", "what": "fax", "due_at": log.t})
    log.add(ET.ESCALATION_OPENED, {"escalation": esc("e1", Track.ORDER, EscalationTrigger.ORDER_STALLED)})
    s.append_many(log.events)
    s.sync_projections(CASE, s.state(CASE))
    s.sync_projections(CASE, s.state(CASE))  # rewrite, no duplicates
    rows = s._db.execute("SELECT cid, status FROM commitments").fetchall()
    assert [tuple(r) for r in rows] == [("c1", "OPEN")]
    assert [e["eid"] for e in s.cached_escalations("OPEN")] == ["e1"]


def test_event_reload_types():
    s = Store()
    s.append_many(happy_path().events)
    evs = s.events(CASE)
    assert all(isinstance(e, Event) and e.sim_time.tzinfo is not None for e in evs)
    assert evs[-1].type == ET.CASE_DONE
