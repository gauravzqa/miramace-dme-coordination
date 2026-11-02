"""B5 voice: offline tests (project venv has no livekit). Live voice is a Mac smoke test only."""

from __future__ import annotations

import importlib.util

import pytest

from miramace.agents.prompts import DISCLOSURE
from miramace.agents.schemas import PersonaTurn
from miramace.calls.brains import FakeCallerBrain, FakePersonaBrain
from miramace.models import CallResult, CallTask
from miramace.voice import transport as vt

HAS_LIVEKIT = importlib.util.find_spec("livekit") is not None


class StubTextPort:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_call(self, task: CallTask) -> CallResult:
        self.calls.append(task.task_id)
        return CallResult(
            task_id=task.task_id,
            outcome="COMPLETED",
            transcript=[],
            claims=[],
            flags=["from_text"],
            duration_s=1,
            callback_after=None,
            transport="text",
        )


def _port(text_port: StubTextPort) -> vt.VoiceCallerPort:
    spec = vt.sample_helpful_maria()
    return vt.VoiceCallerPort(
        text_port,
        lambda task: FakeCallerBrain(),
        lambda pid: FakePersonaBrain(spec),
        lambda pid: spec,
        extractor=None,
    )


def test_module_imports_without_livekit() -> None:
    assert hasattr(vt, "VoiceCallerPort") and hasattr(vt, "run_voice_call") and hasattr(vt, "smoke")


@pytest.mark.skipif(HAS_LIVEKIT, reason="checks the no-livekit fallback path")
async def test_voice_on_without_livekit_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRA_VOICE", "1")
    monkeypatch.delenv("MIRA_VOICE_LEGS", raising=False)
    tp = StubTextPort()
    port = _port(tp)
    res = await port.run_call(vt.sample_pcp_task())
    assert tp.calls == ["smoke-pcp-request-order"]
    assert res.transport == "text_fallback"
    assert "voice_failed:livekit_missing" in res.flags and "from_text" in res.flags
    assert isinstance(port.last_error, ImportError)


async def test_voice_failure_reason_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*a, **k):
        raise vt.VoiceCallError("connect", "refused")

    monkeypatch.setenv("MIRA_VOICE", "1")
    monkeypatch.setattr(vt, "run_voice_call", boom)
    res = await _port(StubTextPort()).run_call(vt.sample_pcp_task())
    assert res.transport == "text_fallback" and res.flags[-1] == "voice_failed:connect"


async def test_voice_off_delegates_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRA_VOICE", "0")
    tp = StubTextPort()
    res = await _port(tp).run_call(vt.sample_pcp_task())
    assert res.transport == "text" and res.flags == ["from_text"] and tp.calls


async def test_supplier_leg_stays_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRA_VOICE", "1")
    monkeypatch.setenv("MIRA_VOICE_LEGS", "pcp")
    task = vt.sample_pcp_task().model_copy(update={"party": "supplier", "party_id": "lakeshore"})
    res = await _port(StubTextPort()).run_call(task)
    assert res.transport == "text"


@pytest.mark.skipif(not HAS_LIVEKIT, reason="livekit.api not installed (voice extra)")
def test_listener_url_builds_offline() -> None:
    url = vt.listener_url("mira-pcp-call")
    assert url.startswith("https://meet.livekit.io/custom?liveKitUrl=ws%3A%2F%2Flocalhost%3A7880&token=")


def test_listener_url_with_given_token() -> None:
    url = vt.listener_url("r", token="abc")
    assert url == "https://meet.livekit.io/custom?liveKitUrl=ws%3A%2F%2Flocalhost%3A7880&token=abc"
    assert "--identity listener" in vt.listener_cli_hint("r")


# ------------------------------------------------------------------ BrainAgent logic via the controller


async def _simulate(ctl: vt.DuplexController, stt=lambda s: s, max_steps: int = 80) -> None:
    """Perfect-turn-taking simulation of the two AgentSessions: llm_node → playout → other side's STT."""
    side = "persona"
    for _ in range(max_steps):
        if ctl.ended.is_set():
            return
        if side == "persona":
            line = await ctl.persona_line()
            if line is None:
                return
            what = ctl.persona_spoke()
            stop = ctl.caller_heard(stt(line))
            if what == "hangup":
                ctl.end("counterparty_hangup")
                return
            side = "persona" if (what == "transfer" or stop and not ctl.ended.is_set()) else "caller"
        else:
            line = await ctl.caller_line()
            if line is None:
                return
            ctl.caller_spoke()
            if ctl.ended.is_set():
                return
            stop = ctl.persona_heard(stt(line))
            side = "caller" if stop else "persona"


async def test_controller_helpful_maria_completes() -> None:
    task, spec = vt.sample_pcp_task(), vt.sample_helpful_maria()
    ctl = vt.DuplexController(task, FakeCallerBrain(), FakePersonaBrain(spec), spec)
    await _simulate(ctl, stt=lambda s: s.lower())  # STT differs from the true text
    raw = ctl.raw()
    assert raw.transport == "voice" and raw.end_reason == "complete"
    utts = raw.utterances
    assert utts[0].speaker == "counterparty" and utts[1].text == DISCLOSURE
    cps = [u for u in utts if u.speaker == "counterparty"]
    assert len(cps) >= 2
    # caller side keeps STT as text and the persona's true line as heard_text
    assert all(u.text == (u.heard_text or "").lower() for u in cps)
    assert all(u.heard_text for u in utts if u.speaker == "caller" and u is not utts[-1])
    # PHI only after verification: the pre-verification caller lines carry no DOB
    assert "1954" not in utts[1].text
    assert ctl.caller_turns <= task.max_turns


async def test_controller_transfer_redisclosure() -> None:
    task = vt.sample_pcp_task()
    spec = vt.sample_helpful_maria().model_copy(update={"behaviour": "transfer"})
    ctl = vt.DuplexController(task, FakeCallerBrain(), FakePersonaBrain(spec), spec)
    await _simulate(ctl)
    texts = [u.text for u in ctl.t_caller.utterances if u.speaker == "caller"]
    assert texts.count(DISCLOSURE) == 2
    assert "transferred" in ctl.flags


class _Hangup:
    async def next(self, view):
        return PersonaTurn(
            say="Sorry, we're closed. Bye.", event="hangup", transfer_to=None, revealed_fact_ids=[]
        )


async def test_controller_persona_hangup_and_nudges() -> None:
    task = vt.sample_pcp_task()
    ctl = vt.DuplexController(task, FakeCallerBrain(), _Hangup(), None)
    assert await ctl.persona_line() == "Sorry, we're closed. Bye."
    assert ctl.caller_heard("sorry we're closed bye") is True
    assert ctl.end_reason == "counterparty_hangup" and await ctl.persona_line() is None

    ctl2 = vt.DuplexController(task, FakeCallerBrain(), _Hangup(), None)
    assert [ctl2.nudge() for _ in range(3)] == [vt.NUDGE_TEXT, vt.NUDGE_TEXT, vt.GIVE_UP_TEXT]
    ctl2.caller_spoke()
    assert ctl2.end_reason == "no_answer"


class _Broken:
    async def next(self, view):
        raise RuntimeError("llm down")


async def test_controller_brain_error_fails_call() -> None:
    task = vt.sample_pcp_task()
    ctl = vt.DuplexController(task, FakeCallerBrain(), _Broken(), None)
    assert await ctl.persona_line() is None
    assert ctl.ended.is_set() and ctl.error == "persona_brain:RuntimeError"
