"""B7 — CLI (typer CliRunner) and render smoke tests. Engine-dependent tests skip until B6 lands."""

from __future__ import annotations

import importlib
import re
from datetime import timedelta

import pytest
from rich.console import Console
from typer.testing import CliRunner

from miramace import render
from miramace.cli import app
from miramace.clock import SIM_START
from miramace.events import EventType, Source, make_event
from miramace.models import (
    Address,
    CallResult,
    Claim,
    CostInfo,
    Escalation,
    EscalationLane,
    EscalationOption,
    EscalationTrigger,
    LCDChecklist,
    LCDCriterion,
    Snapshot,
    SnapshotItem,
    SupplierFacts,
    Track,
    Utterance,
    dr_chen,
    eleanor,
)
from miramace.store import TranscriptRecord

runner = CliRunner()
LAKE = "lakeshore-home-medical-equipment"


def _engine_ready() -> bool:
    try:
        mod = importlib.import_module("miramace.scenario")
    except Exception:
        return False
    return hasattr(mod, "run_scenario") and hasattr(importlib.import_module("miramace.metrics"), "score")


def _con() -> Console:
    return Console(record=True, width=140, force_terminal=False)


def sample_events():
    t = SIM_START
    rule = Source(kind="RULE", ref="planner")
    tr = Source(kind="TRANSCRIPT", ref="tr-1")
    esc = Escalation(
        eid="esc-cost_concern-1",
        lane=EscalationLane.PATIENT,
        trigger=EscalationTrigger.COST_CONCERN,
        track=Track.PATIENT,
        question="Eleanor raised a cost concern",
        suggested="nurse_resolved_accepted",
        options=[EscalationOption(option_id="nurse_resolved_accepted", label="accepts")],
    )
    specs = [
        (EventType.CASE_OPENED, {"patient": eleanor(), "pcp": dr_chen()}, Source(kind="SYSTEM", ref="x"), 0),
        (
            EventType.ADDRESS_CONFIRMED,
            {"address": Address.parse("2150 N Lincoln Park W, Chicago, IL 60614")},
            Source(kind="PATIENT_INTENT", ref="c1"),
            1,
        ),
        (EventType.CANDIDATES_RANKED, {"sids": [LAKE], "distances": {LAKE: 0.6}}, rule, 2),
        (
            EventType.SUPPLIER_QUALIFIED,
            {
                "sid": LAKE,
                "verified_at": t,
                "facts": SupplierFacts(
                    k0001_in_stock=True,
                    earliest_delivery_days=2,
                    evidence={"k0001_in_stock": "Yes, we have standard K0001 chairs in stock."},
                ),
            },
            tr,
            3,
        ),
        (
            EventType.COMMITMENT_MADE,
            {
                "cid": "c1",
                "party": "pcp",
                "what": "fax signed order",
                "due_at": t + timedelta(days=3),
                "evidence": "By Thursday.",
            },
            tr,
            4,
        ),
        (
            EventType.LCD_EVALUATED,
            {
                "checklist": LCDChecklist(results={LCDCriterion.MRADL_LIMITATION: "met"}),
                "missing": [LCDCriterion.HOME_ACCOMMODATES],
            },
            Source(kind="INBOUND_DOC", ref="d"),
            30,
        ),
        (EventType.ESCALATION_OPENED, {"escalation": esc}, rule, 31),
        (
            EventType.HUMAN_RESOLVED,
            {"eid": esc.eid, "option_id": "nurse_resolved_accepted", "resolver": "advocate"},
            Source(kind="HUMAN", ref=esc.eid),
            50,
        ),
        (EventType.CASE_DONE, None, rule, 100),
    ]
    return [make_event("case-x", et, p, s, t + timedelta(hours=h)) for et, p, s, h in specs]


def test_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "miramace" in res.stdout


def test_every_event_type_has_a_track() -> None:
    missing = [
        t
        for t in EventType
        if t not in render.TRACK_BY_TYPE and t not in (EventType.ESCALATION_OPENED, EventType.HUMAN_RESOLVED)
    ]
    assert not missing


def test_render_timeline_and_rows() -> None:
    evs = sample_events()
    row = render.event_row(evs[3])
    assert row[0] == "Mon 10:00" or re.match(r"\w{3} \d\d:\d\d", row[0])
    assert row[1] == "supplier" and row[3] == "TRANSCRIPT"
    assert "K0001 chairs in stock" in row[4]
    con = _con()
    n = render.render_timeline(evs, console=con)
    out = con.export_text()
    assert n == len(evs)
    assert "Monday, Nov 2" in out and "ESCALATE COST_CONCERN" in out
    con2 = _con()
    assert render.render_timeline(evs, track="order", console=con2) >= 2  # LCD + ledger
    printer = render.live_printer(con2)
    for e in evs:
        printer(e)


def test_render_panels_do_not_crash() -> None:
    con = _con()
    snap = Snapshot(
        case_id="case-x",
        as_of=SIM_START,
        headline="Waiting on Dr. Chen's office.",
        items=[
            SnapshotItem(
                track=Track.ORDER,
                plain_status="Order requested.",
                owner_label="Dr. Chen's office",
                due_label="Thursday, Nov 5",
            )
        ],
        cost=CostInfo(),
        next_patient_action="Nothing right now.",
    )
    render.render_snapshot(snap, console=con)
    result = CallResult(
        task_id="t1",
        outcome="COMPLETED",
        flags=["readback_ok"],
        duration_s=90,
        callback_after=None,
        transport="fake",
        transcript=[
            Utterance(turn=0, speaker="counterparty", name="Maria", text="Sunrise, this is Maria.", ts=0),
            Utterance(turn=1, speaker="caller", name="Mira", text="Hi.", ts=1, event="hangup"),
        ],
        claims=[Claim(field="promised_by", status="stated", value="Thursday", evidence_quote="By Thursday.")],
    )
    rec = TranscriptRecord(
        transcript_id="tr-1",
        case_id="case-x",
        task_id="case-x:pcp:request_order:v1",
        party_id="pcp",
        sim_time=SIM_START,
        result=result,
    )
    render.render_transcript(rec, console=con)
    render.render_transcript(result, console=con)
    render.render_transcript_list([rec], console=con)
    esc = sample_events()[6].parsed().escalation
    render.render_escalations([esc], console=con)
    render.render_escalations([], console=con)

    class S:  # duck-typed RunSummary / score / metrics
        case_id, scenario, status = "case-x", "happy_path", "DONE"
        sim_start, sim_end, steps, calls = SIM_START, SIM_START + timedelta(days=5), 40, 12
        escalations, open_escalations, invariant_violations = [], [], []
        usage_tokens, cost_usd = 0, None

    render.render_summary(S(), {"passed": True, "false_close": False}, console=con)
    render.render_metrics(
        {
            "scenarios": 1,
            "autonomous_closes": 1,
            "false_closes": 0,
            "escalation_precision": 1.0,
            "escalation_recall": 1.0,
            "invariant_violations": 0,
            "phi_leaks": 0,
            "all_passed": True,
        },
        [{"scenario": "happy_path", "final": "DONE", "expected_final": "DONE", "passed": True}],
        console=con,
    )
    out = con.export_text()
    assert "By Thursday." in out and "ALL PASSED" in out


def test_call_pcp_text_fake() -> None:
    res = runner.invoke(app, ["call", "pcp", "--text", "--llm", "fake"])
    assert res.exit_code == 0, res.stdout
    assert "Maria" in res.stdout and "promised_by" in res.stdout


def test_live_mode_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False, raising=False)
    res = runner.invoke(app, ["call", "pcp", "--text", "--llm", "live"])
    assert res.exit_code == 2
    assert "OPENAI_API_KEY" in res.stdout


@pytest.mark.skipif(not _engine_ready(), reason="B6 engine not available yet")
def test_run_happy_path_quiet(tmp_path) -> None:
    res = runner.invoke(app, ["run", "happy_path", "--quiet", "--db-dir", str(tmp_path)])
    assert res.exit_code == 0, res.stdout
    m = re.search(r"case_id: (\S+)", res.stdout)
    assert m and "DONE" in res.stdout
    case_id = m.group(1)
    for args in (
        ["timeline", case_id],
        ["snapshot", case_id],
        ["transcripts", case_id],
        ["escalations", case_id],
        ["timeline", case_id, "--track", "supplier"],
    ):
        r = runner.invoke(app, [*args, "--db-dir", str(tmp_path)])
        assert r.exit_code == 0, (args, r.stdout)


@pytest.mark.skipif(not _engine_ready(), reason="B6 engine not available yet")
def test_resolve_cost_concern_resumes_to_done(tmp_path) -> None:
    res = runner.invoke(app, ["run", "cost_concern", "--quiet", "--db-dir", str(tmp_path)])
    assert res.exit_code == 0, res.stdout
    case_id = re.search(r"case_id: (\S+)", res.stdout).group(1)
    r = runner.invoke(
        app,
        [
            "resolve",
            case_id,
            "esc-cost_concern-1",
            "nurse_resolved_accepted",
            "--note",
            "nurse called; she is fine with it",
            "--db-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert "DONE" in r.stdout
