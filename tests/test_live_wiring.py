"""LIVE (OpenAI) wiring, exercised offline.

* every scenario through build_ports(mode="live") with StubResponsesLLM in place of OpenAILLM;
* OpenAILLM through the REAL openai SDK (AsyncOpenAI + httpx.MockTransport): request body + parsing;
* repair retry -> LLMUnavailable -> sane outcomes; call budget; CLI env handling.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from live_stub import StubResponsesLLM, install  # noqa: E402
from miramace.agents.schemas import CallerTurn, CompanionOut, Extraction, LCDCheck, PersonaTurn  # noqa: E402
from miramace.calls.brains import to_messages  # noqa: E402
from miramace.events import EventType  # noqa: E402
from miramace.llm import port as llm_port  # noqa: E402
from miramace.llm.port import LLMUnavailable, OpenAILLM, cost_usd  # noqa: E402
from miramace.models import Utterance  # noqa: E402
from miramace.scenario import list_scenarios, load_scenario, run_scenario  # noqa: E402

httpx = pytest.importorskip("httpx2")
openai = pytest.importorskip("openai")

SCENARIOS = list_scenarios()


def _run(name: str, mode: str = "live"):
    return asyncio.run(run_scenario(name, mode=mode, db_dir=None))


# --------------------------------------------------------------------------------- all scenarios


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_live_wiring_reaches_expected_final(monkeypatch, name):
    sc = load_scenario(name)
    stub = install(monkeypatch, StubResponsesLLM(sc))
    summary, store, _ = _run(name)
    assert stub.errors == []
    assert summary.status == sc.expect.final
    assert sorted(summary.escalations) == sorted(sc.expect.escalations)
    assert summary.invariant_violations == []

    by_agent = stub.calls_by_agent()
    assert by_agent["caller"] > 0 and by_agent["persona"] > 0 and by_agent["extractor"] > 0
    assert by_agent["lcd"] >= 1  # live doc reader is wired
    # every patient notification was worded by the LLM companion (and passed grounding)
    notified = [e for e in store.events(summary.case_id) if e.type == EventType.PATIENT_NOTIFIED]
    compose = [r for r in stub.requests if r["agent"] == "companion" and r["key"].startswith("compose")]
    assert len(compose) >= len(notified) > 0
    assert any("quick update about your wheelchair" in e.payload["text"] for e in notified)
    # extraction pass A + B per answered call; LLM-backed claims reached the domain
    completed = [t for t in store.transcripts(summary.case_id) if t.result.outcome == "COMPLETED"]
    assert by_agent["extractor"] == 2 * len(
        [t for t in store.transcripts(summary.case_id) if t.result.outcome not in ("NO_ANSWER", "REFUSED")]
    )
    assert any(c.status == "stated" for t in completed for c in t.result.claims)
    # usage + cost aggregated into the summary from the run's ledger
    assert summary.usage_tokens == sum(c.in_tok + c.out_tok for c in stub.ledger) > 0
    assert summary.cost_usd == cost_usd(stub.ledger) and summary.cost_usd > 0
    # per-call usage recorded on transcripts too
    assert sum(len(t.result.usage) for t in completed) > 0
    assert len(stub.requests) < 400  # well inside the default MIRA_MAX_LLM_CALLS budget


def test_live_matches_fake_call_count(monkeypatch):
    fake, _, _ = _run("happy_path", mode="fake")
    install(monkeypatch, StubResponsesLLM(load_scenario("happy_path")))
    live, _, _ = _run("happy_path")
    assert (live.status, live.calls, live.escalations) == (fake.status, fake.calls, fake.escalations)


def test_cli_run_live_with_stub(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from miramace.cli import app

    stub = install(monkeypatch, StubResponsesLLM(load_scenario("happy_path")))
    res = CliRunner().invoke(app, ["run", "happy_path", "--llm", "live", "-q", "--db-dir", str(tmp_path)])
    assert res.exit_code == 0, res.stdout
    assert "DONE" in res.stdout and "tokens" in res.stdout
    assert stub.errors == [] and len(stub.requests) > 50


# --------------------------------------------------------------------------- OpenAILLM via the SDK


def _response_json(text: str, *, in_tok=11, out_tok=7, status="completed") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "dummy-model",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _sdk_llm(texts: list[str], **kw) -> tuple[OpenAILLM, list[dict]]:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_response_json(texts[min(len(bodies) - 1, len(texts) - 1)]))

    client = openai.AsyncOpenAI(
        api_key="sk-dummy",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAILLM("dummy-model", client=client, **kw), bodies


def _transcript() -> list[Utterance]:
    return [
        Utterance(
            turn=0, speaker="counterparty", name="Maria", text="Sunrise Family Medicine, this is Maria.", ts=0
        ),
        Utterance(turn=1, speaker="caller", name="Mira Mace AI", text="Hi, this is an AI assistant.", ts=6),
        Utterance(
            turn=2, speaker="counterparty", name="Maria", text="Okay, how can I help?", ts=12, event="hold"
        ),
    ]


@pytest.mark.parametrize("effort", [None, "low"])
async def test_openai_llm_request_shape_through_sdk(monkeypatch, effort):
    if effort:
        monkeypatch.setenv("MIRA_REASONING_EFFORT", effort)
    else:
        monkeypatch.delenv("MIRA_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("MIRA_MAX_OUTPUT_TOKENS", raising=False)
    good = CallerTurn(say="Have I reached Sunrise?", slots_touched=[], action="continue", hangup_reason=None)
    llm, bodies = _sdk_llm([good.model_dump_json()])
    msgs = to_messages(_transcript(), "caller")
    out = await llm.structured(
        agent="caller", key="t1:t3", system="SYS PROMPT", messages=msgs, schema=CallerTurn
    )
    assert out == good and isinstance(out, CallerTurn)
    body = bodies[0]
    assert body["model"] == "dummy-model" and body["instructions"] == "SYS PROMPT"
    assert body["store"] is False and body["max_output_tokens"] == llm_port.DEFAULT_MAX_OUTPUT_TOKENS
    assert body["input"] == msgs
    assert all(set(m) == {"role", "content"} and m["role"] in ("user", "assistant") for m in body["input"])
    assert body["input"][-1]["role"] == "user"  # the model always answers a user turn
    fmt = body["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "CallerTurn"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == set(CallerTurn.model_fields)
    if effort:
        assert body["reasoning"] == {"effort": "low"}
    else:
        assert "reasoning" not in body
    assert "temperature" not in body
    (call,) = llm.ledger
    assert (call.agent, call.key, call.model, call.in_tok, call.out_tok, call.ok) == (
        "caller",
        "t1:t3",
        "dummy-model",
        11,
        7,
        True,
    )


@pytest.mark.parametrize("schema", [CallerTurn, PersonaTurn, Extraction, LCDCheck, CompanionOut])
async def test_every_agent_schema_round_trips_through_sdk(schema):
    from live_stub import AGENT_SCHEMAS

    samples = {
        CallerTurn: CallerTurn(say="x", slots_touched=["a"], action="hangup", hangup_reason="complete"),
        PersonaTurn: PersonaTurn(say="x", event="transfer", transfer_to="Jordan", revealed_fact_ids=[]),
        Extraction: Extraction.model_validate(
            {"claims": [{"field": "f", "status": "unknown", "value": None, "evidence_quote": None}]}
        ),
        LCDCheck: LCDCheck.model_validate(
            {"results": [{"criterion_id": "home_accommodates", "met": "met", "quote": "34-inch doorways"}]}
        ),
        CompanionOut: CompanionOut.model_validate(
            {
                "reply": "ok",
                "intents": [{"kind": "needs_human", "address": None, "preferred_time": None, "note": "x"}],
            }
        ),
    }
    agent = next(a for a, s in AGENT_SCHEMAS.items() if s is schema)
    llm, bodies = _sdk_llm([samples[schema].model_dump_json()])
    out = await llm.structured(
        agent=agent, key="k", system="s", messages=[{"role": "user", "content": "x"}], schema=schema
    )
    assert out == samples[schema]
    assert bodies[0]["text"]["format"]["strict"] is True


async def test_parse_failure_repairs_once_then_unavailable(monkeypatch):
    monkeypatch.delenv("MIRA_MAX_OUTPUT_TOKENS", raising=False)
    llm, bodies = _sdk_llm(['{"say": "truncated'])  # invalid JSON every time
    with pytest.raises(LLMUnavailable, match="after repair"):
        await llm.structured(
            agent="caller",
            key="k",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            schema=CallerTurn,
        )
    assert len(bodies) == 2  # first try + exactly one repair
    assert len(bodies[1]["input"]) == 2 and "invalid" in bodies[1]["input"][-1]["content"]
    assert bodies[1]["max_output_tokens"] == 4 * bodies[0]["max_output_tokens"]
    assert [c.ok for c in llm.ledger] == [False, False]

    llm2, bodies2 = _sdk_llm(
        [
            '{"say": "trunc',
            CallerTurn(say="ok", slots_touched=[], action="continue", hangup_reason=None).model_dump_json(),
        ]
    )
    out = await llm2.structured(
        agent="caller", key="k", system="s", messages=[{"role": "user", "content": "x"}], schema=CallerTurn
    )
    assert out.say == "ok" and [c.ok for c in llm2.ledger] == [False, True]


async def test_api_error_maps_to_unavailable():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad model", "type": "invalid_request_error"}})

    client = openai.AsyncOpenAI(
        api_key="sk-dummy",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    llm = OpenAILLM("nope", client=client)
    with pytest.raises(LLMUnavailable, match="BadRequestError"):
        await llm.structured(
            agent="caller",
            key="k",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            schema=CallerTurn,
        )
    assert len(llm.ledger) == 1


def test_default_client_has_timeout_and_retries(monkeypatch):
    seen: dict = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.delenv("MIRA_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("MIRA_LLM_RETRIES", raising=False)
    OpenAILLM("m").client  # noqa: B018
    assert seen == {"timeout": llm_port.DEFAULT_TIMEOUT_S, "max_retries": llm_port.DEFAULT_SDK_RETRIES}
    monkeypatch.setenv("MIRA_LLM_TIMEOUT", "20")
    monkeypatch.setenv("MIRA_LLM_RETRIES", "5")
    OpenAILLM("m").client  # noqa: B018
    assert seen == {"timeout": 20.0, "max_retries": 5}


async def test_call_budget(monkeypatch):
    good = CallerTurn(say="x", slots_touched=[], action="continue", hangup_reason=None).model_dump_json()
    llm, bodies = _sdk_llm([good], max_calls=2)
    kw = dict(
        agent="caller", key="k", system="s", messages=[{"role": "user", "content": "x"}], schema=CallerTurn
    )
    await llm.structured(**kw)
    await llm.structured(**kw)
    with pytest.raises(LLMUnavailable, match="budget"):
        await llm.structured(**kw)
    assert len(bodies) == 2  # the network was not touched
    monkeypatch.setenv("MIRA_MAX_LLM_CALLS", "7")
    assert OpenAILLM("m").max_calls == 7
    monkeypatch.delenv("MIRA_MAX_LLM_CALLS")
    assert OpenAILLM("m").max_calls == llm_port.DEFAULT_MAX_LLM_CALLS == 400


# ------------------------------------------------------------------------- failures -> sane outcomes


def test_extractor_unavailable_gives_unknown_claims_not_crash(monkeypatch):
    stub = install(monkeypatch, StubResponsesLLM(load_scenario("happy_path"), fail_agents={"extractor"}))
    summary, store, _ = _run("happy_path")
    trs = store.transcripts(summary.case_id)
    assert trs and summary.status in ("STUCK", "BLOCKED_ON_HUMAN", "MAX_STEPS")
    answered = [t for t in trs if t.result.outcome not in ("NO_ANSWER", "REFUSED")]
    assert answered
    for t in answered:
        assert all(c.status == "unknown" for c in t.result.claims)
        assert any(f.startswith("extraction_failed:LLMUnavailable") for f in t.result.flags)
    assert summary.invariant_violations == []
    assert not stub.errors


def test_persona_unavailable_gives_failed_calls(monkeypatch):
    install(monkeypatch, StubResponsesLLM(load_scenario("happy_path"), fail_agents={"persona"}))
    summary, store, _ = _run("happy_path")
    outs = {t.result.outcome for t in store.transcripts(summary.case_id)}
    assert "FAILED" in outs and outs <= {"FAILED", "NO_ANSWER"}  # NO_ANSWER = sim suppliers w/o persona
    assert summary.status != "DONE" and summary.invariant_violations == []


def test_live_run_with_broken_model_through_openai_llm(monkeypatch):
    """Real OpenAILLM wiring (make_llm -> build_ports), model output never parses: every request is
    repaired once, the run ends cleanly, nothing crashes, the budget bounds the damage."""
    texts = ["not json"]
    real_make = llm_port.make_llm

    def make(mode=None):
        llm = real_make("live")
        llm._client, _ = _sdk_llm(texts)[0]._client, None
        return llm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("MIRA_TEXT_MODEL", "dummy-model")
    monkeypatch.setenv("MIRA_MAX_LLM_CALLS", "60")
    monkeypatch.setattr(llm_port, "make_llm", make)
    holder: dict = {}
    import miramace.scenario as scen

    orig = scen.build_ports

    def bp(*a, **k):
        holder["ports"] = orig(*a, **k)
        return holder["ports"]

    monkeypatch.setattr(scen, "build_ports", bp)
    summary, store, _ = _run("happy_path")
    llm = holder["ports"].llm
    assert isinstance(llm, OpenAILLM) and llm.requests_made <= 60
    assert all(not c.ok for c in llm.ledger)
    assert {t.result.outcome for t in store.transcripts(summary.case_id)} <= {"FAILED", "NO_ANSWER"}
    assert summary.status != "DONE" and summary.invariant_violations == []
    assert summary.usage_tokens == sum(c.in_tok + c.out_tok for c in llm.ledger)


# ------------------------------------------------------------------------------------------ CLI env


def test_cli_live_reports_missing_model(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from miramace.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.delenv("MIRA_TEXT_MODEL", raising=False)
    res = CliRunner().invoke(app, ["run", "happy_path", "--llm", "live", "-q"])
    assert res.exit_code == 2
    out = re.sub(r"\s+", " ", res.stdout)
    assert "MIRA_TEXT_MODEL" in out and "OPENAI_API_KEY" not in out.split("needs")[1].split(".")[0]


def test_cli_loads_dotenv_from_cwd_then_repo_root(monkeypatch, tmp_path):
    from miramace import cli

    root = tmp_path / "repo"
    app_dir = tmp_path / "app"
    root.mkdir()
    app_dir.mkdir()
    (root / ".env").write_text("MIRA_T_A=root\nMIRA_T_B=root\n")
    (app_dir / ".env").write_text("MIRA_T_A=cwd\n")
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    monkeypatch.chdir(app_dir)
    for k in ("MIRA_T_A", "MIRA_T_B"):
        monkeypatch.delenv(k, raising=False)
    try:
        files = cli.load_env()
        assert files == [app_dir / ".env", root / ".env"]
        assert os.environ["MIRA_T_A"] == "cwd" and os.environ["MIRA_T_B"] == "root"
    finally:
        os.environ.pop("MIRA_T_A", None)
        os.environ.pop("MIRA_T_B", None)


def test_cli_max_llm_calls_sets_env(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from miramace.cli import app

    monkeypatch.setenv("MIRA_MAX_LLM_CALLS", "400")
    res = CliRunner().invoke(
        app, ["run", "happy_path", "-q", "--db-dir", str(tmp_path), "--max-llm-calls", "123"]
    )
    assert res.exit_code == 0, res.stdout
    assert os.environ["MIRA_MAX_LLM_CALLS"] == "123"


# ------------------------------------------------------------------------------------------ voice leg


@pytest.mark.parametrize("mode", ["fake", "live"])
def test_build_ports_voice_pcp_wraps_pcp_leg(monkeypatch, mode):
    from miramace.calls.brains import FakeCallerBrain, LLMCallerBrain
    from miramace.calls.runner import TextCallerPort
    from miramace.clock import SIM_START, VirtualClock
    from miramace.scenario import build_ports
    from miramace.store import Store
    from miramace.voice.transport import VoiceCallerPort

    monkeypatch.setenv(
        "MIRA_VOICE", "0"
    )  # --voice-pcp overrides a .env default of 0 (restored by monkeypatch)
    monkeypatch.setenv("MIRA_VOICE_LEGS", "supplier")
    sc = load_scenario("happy_path")
    if mode == "live":
        install(monkeypatch, StubResponsesLLM(sc))
    ports = build_ports(sc, Store(":memory:"), VirtualClock(SIM_START), mode=mode, voice_pcp=True)
    voice = ports.caller_pcp.inner
    assert isinstance(voice, VoiceCallerPort)
    assert isinstance(voice.text_port, TextCallerPort)
    assert not isinstance(ports.caller_supplier.inner, VoiceCallerPort)
    assert os.environ["MIRA_VOICE"] == "1" and "pcp" in os.environ["MIRA_VOICE_LEGS"].split(",")
    brain = voice.caller_brain_factory()
    assert isinstance(brain, LLMCallerBrain if mode == "live" else FakeCallerBrain)
    assert voice.persona_lookup("pcp").answerer_name == "Maria"
    assert voice.persona_brain_factory("pcp").spec.org_name == "Sunrise Family Medicine"


def test_voice_pcp_without_livekit_falls_back_to_text(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("livekit") is not None:
        pytest.skip("livekit installed: this test covers the no-voice-extra fallback")
    monkeypatch.setenv("MIRA_VOICE", "1")
    monkeypatch.setenv("MIRA_VOICE_LEGS", "pcp")
    summary, store, _ = asyncio.run(run_scenario("happy_path", mode="fake", voice_pcp=True, db_dir=None))
    assert summary.status == "DONE"
    trs = store.transcripts(summary.case_id)
    pcp = [t for t in trs if t.result.task_id and t.party_id == "pcp"]
    sup = [t for t in trs if t.party_id != "pcp"]
    assert pcp and all(t.result.transport == "text_fallback" for t in pcp)
    assert all(any(f.startswith("voice_failed:") for f in t.result.flags) for t in pcp)
    assert sup and all(t.result.transport != "text_fallback" for t in sup)
