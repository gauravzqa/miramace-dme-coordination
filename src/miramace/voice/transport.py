"""B5 Voice — VoiceCallerPort: two in-process AgentSessions + automatic text fallback.

Layout
- `DuplexController` (pure Python, no livekit): owns BOTH perspectives of one call and drives the SAME
  `ports.Brain`s the text runner uses. Caller-side transcript = caller true text + STT of the persona
  (persona true text kept as `heard_text`); persona-side transcript = STT of the caller + persona true text.
  Phase / disclosure / guards come from `calls.runner` (phase_from, needs_disclosure, guard_caller_turn).
- `BrainAgent` (created by `make_brain_agent_cls()` only when livekit imports): a thin livekit `Agent` whose
  `on_user_turn_completed` feeds STT into the controller and whose `llm_node` returns the brain's line.
- `run_voice_call`: two `rtc.Room` connections in one process ("mira-caller" and f"{party}-frontdesk"), two
  `AgentSession`s pinned to each other, VAD turn detection, interruptions off, 20 s silence watchdog,
  overall timeout, clean teardown → `calls.runner.RawCall` (transport="voice").
- `VoiceCallerPort`: CallerPort; voice when MIRA_VOICE=1 and the party is in MIRA_VOICE_LEGS, else / on any
  failure → the text port (transport="text_fallback", flag `voice_failed:<reason>`).

Verified against livekit-agents 1.8.10 / livekit-plugins-openai 1.8.10 source (see docs/voice-smoke.md).
Every livekit import is inside a function so this module imports without the `voice` extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import re
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from typing import Any, Literal

from miramace.agents.schemas import CallerTurn, PersonaTurn
from miramace.models import CallResult, CallTask, LLMCall, PersonaSpec, slugify

log = logging.getLogger("miramace.voice")

CALLER_IDENTITY = "mira-caller"
LISTENER_IDENTITY = "listener"
MEET_URL = "https://meet.livekit.io/custom"
SILENCE_S = 20.0
MAX_NUDGES = 2
NUDGE_TEXT = "Hello, are you still there?"
GIVE_UP_TEXT = "It sounds like we got disconnected. We'll try again later. Goodbye."
MAX_TURNS_TEXT = "Thank you for your time. I'll pass this to the care team. Goodbye."
MIN_COUNTERPARTY_UTTERANCES = 2

Side = Literal["caller", "persona"]


class VoiceCallError(RuntimeError):
    """Voice leg could not produce a usable call; `reason` becomes the `voice_failed:<reason>` flag."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# ------------------------------------------------------------------------------------------ env


def lk_env() -> tuple[str, str, str]:
    return (
        os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
        os.environ.get("LIVEKIT_API_KEY", "devkey"),
        os.environ.get("LIVEKIT_API_SECRET", "secret"),
    )


def persona_identity(task: CallTask) -> str:
    return f"{task.party}-frontdesk"


def room_name(task: CallTask) -> str:
    return os.environ.get("MIRA_DEMO_ROOM") or f"call-{slugify(task.task_id)}"


def voice_enabled_for(task: CallTask) -> bool:
    if os.environ.get("MIRA_VOICE", "0").strip() != "1":
        return False
    legs = {p.strip() for p in os.environ.get("MIRA_VOICE_LEGS", "pcp").split(",") if p.strip()}
    return task.party in legs


# --------------------------------------------------------------------------------------- tokens


def access_token(identity: str, room: str, *, publish: bool = True, hidden: bool = False) -> str:
    """JWT for local livekit-server. livekit.api.AccessToken(...).with_identity().with_grants(VideoGrants)."""
    from livekit import api  # lazy: voice extra only

    _, key, secret = lk_env()
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=publish,
        can_subscribe=True,
        can_publish_data=publish,
        hidden=hidden,
    )
    return (
        api.AccessToken(key, secret).with_identity(identity).with_name(identity).with_grants(grants).to_jwt()
    )


def listener_token(room: str) -> str:
    """Hidden, subscribe-only token for a human listener."""
    return access_token(LISTENER_IDENTITY, room, publish=False, hidden=True)


def listener_url(room: str, token: str | None = None) -> str:
    url, _, _ = lk_env()
    tok = token if token is not None else listener_token(room)
    return f"{MEET_URL}?{urllib.parse.urlencode({'liveKitUrl': url, 'token': tok})}"


def listener_cli_hint(room: str) -> str:
    url, key, secret = lk_env()
    return (
        f"lk token create --url {url} --api-key {key} --api-secret {secret} "
        f"--join --room {room} --identity {LISTENER_IDENTITY} --valid-for 1h"
    )


# ----------------------------------------------------------------------------------- controller


def _add_flag(flags: list[str], f: str) -> None:
    if f not in flags:
        flags.append(f)


class DuplexController:
    """Turn bookkeeping for one voice call; no livekit dependency (unit-tested with stub STT).

    The livekit side calls: persona_line()/caller_line() from llm_node; caller_heard()/persona_heard() from
    on_user_turn_completed; caller_spoke()/persona_spoke() when an assistant item finished; nudge() from the
    watchdog. `ended` is set when the call is over (`end_reason`) or failed (`error`).
    """

    def __init__(
        self,
        task: CallTask,
        caller_brain: Any,
        persona_brain: Any,
        persona_spec: PersonaSpec | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from miramace.calls.runner import Transcript

        self.task = task
        self.caller_brain = caller_brain
        self.persona_brain = persona_brain
        self.spec = persona_spec
        self.clock = clock
        self.t0 = clock()
        self.t_caller = Transcript()  # caller perspective (finalized)
        self.t_persona = Transcript()  # persona perspective (persona brain view)
        self.flags: list[str] = []
        self.cp_name = persona_spec.answerer_name if persona_spec else task.org_name
        if persona_spec is not None and persona_spec.behaviour == "ivr":
            self.cp_name = "Phone menu"
        self._pending_persona: list[tuple[PersonaTurn, str]] = []  # spoken, not yet heard by caller
        self._unheard_caller: list[int] = []  # t_caller indices the persona has not heard yet
        self.caller_turns = 0
        self.nudges = 0
        self.pending_end: str | None = None  # caller said its last line; end once played out
        self.persona_hung_up = False
        self.transfer_pending = False
        self.end_reason: str | None = None
        self.error: str | None = None
        self.ended = asyncio.Event()
        self.last_activity = clock()

    # -------------------------------------------------------------- helpers
    def _ts(self) -> float:
        return round(self.clock() - self.t0, 2)

    def touch(self) -> None:
        self.last_activity = self.clock()

    def end(self, reason: str) -> None:
        if self.end_reason is None and self.error is None:
            self.end_reason = reason
        self.ended.set()

    def fail(self, reason: str) -> None:
        if self.error is None and self.end_reason is None:
            self.error = reason
        self.ended.set()

    def phase(self) -> str:
        from miramace.calls.runner import phase_from

        return phase_from(self.t_caller, self.task)

    def counterparty_count(self) -> int:
        return sum(1 for u in self.t_caller.utterances if u.speaker == "counterparty")

    # -------------------------------------------------------------- lines (llm_node)
    async def caller_line(self) -> str | None:
        from miramace.agents.prompts import DISCLOSURE
        from miramace.calls.runner import guard_caller_turn, needs_disclosure

        if self.ended.is_set() or self.pending_end is not None:
            return None
        self.touch()
        if self.caller_turns >= self.task.max_turns:
            _add_flag(self.flags, "max_turns")
            ct = CallerTurn(say=MAX_TURNS_TEXT, slots_touched=[], action="hangup", hangup_reason="max_turns")
        elif needs_disclosure(self.t_caller):
            ct = CallerTurn(say=DISCLOSURE, slots_touched=[], action="continue", hangup_reason=None)
        else:
            phase = self.phase()
            try:
                out = await self.caller_brain.next(self.t_caller.view(self.task, phase, "caller"))
            except Exception as e:  # brain failure → voice leg fails → text fallback
                log.exception("caller brain failed")
                self.fail(f"caller_brain:{type(e).__name__}")
                return None
            if not isinstance(out, CallerTurn):
                self.fail("caller_brain:bad_type")
                return None
            ct, gflags = guard_caller_turn(self.task, phase, out)
            for f in gflags:
                _add_flag(self.flags, f)
        self.caller_turns += 1
        self.t_caller.add_caller_turn(ct, ts=self._ts())
        self._unheard_caller.append(len(self.t_caller.utterances) - 1)
        if ct.action == "hangup":
            self.pending_end = ct.hangup_reason or "complete"
        return ct.say

    async def persona_line(self) -> str | None:
        from miramace.extract.extractor import looks_like_injection

        if self.ended.is_set() or self.persona_hung_up:
            return None
        self.touch()
        view = self.t_persona.view(self.task, self.phase(), "counterparty", self.spec)
        try:
            pt = await self.persona_brain.next(view)
        except Exception as e:
            log.exception("persona brain failed")
            self.fail(f"persona_brain:{type(e).__name__}")
            return None
        if not isinstance(pt, PersonaTurn):
            self.fail("persona_brain:bad_type")
            return None
        name = self.cp_name
        self.t_persona.add_persona_turn(pt, name, ts=self._ts())
        self._pending_persona.append((pt, name))
        if looks_like_injection(pt.say):
            _add_flag(self.flags, "injection_suspected")
        if pt.event == "transfer":
            _add_flag(self.flags, "transferred")
            self.cp_name = pt.transfer_to or "Transferred party"
            self.transfer_pending = True
        if pt.event == "hangup":
            self.persona_hung_up = True
        return pt.say

    # -------------------------------------------------------------- hearing (on_user_turn_completed)
    def caller_heard(self, stt_text: str) -> bool:
        """Caller's STT of the persona. Returns True when the caller must NOT reply."""
        self.touch()
        self.nudges = 0
        pend, self._pending_persona = self._pending_persona, []
        events = [pt.event for pt, _ in pend]
        event = next((e for e in ("hangup", "transfer", "hold") if e in events), None)
        name = pend[-1][1] if pend else self.cp_name
        heard = " ".join(pt.say for pt, _ in pend) or None
        self.t_caller.add("counterparty", name, stt_text, event=event, heard_text=heard, ts=self._ts())
        if self.persona_hung_up and not self._pending_persona:
            self.end("counterparty_hangup")
            return True
        if pend and pend[-1][0].event == "transfer":  # the new person is about to speak; wait for them
            return True
        return self.ended.is_set() or self.pending_end is not None

    def persona_heard(self, stt_text: str) -> bool:
        """Persona's STT of the caller. Returns True when the persona must NOT reply."""
        self.touch()
        for i in self._unheard_caller:
            u = self.t_caller.utterances[i]
            self.t_caller.utterances[i] = u.model_copy(update={"heard_text": stt_text})
        self._unheard_caller = []
        self.t_persona.add("caller", "Caller", stt_text, ts=self._ts())
        return self.ended.is_set() or self.persona_hung_up or self.pending_end is not None

    # -------------------------------------------------------------- playout
    def caller_spoke(self) -> None:
        self.touch()
        if self.pending_end is not None:
            self.end(self.pending_end)

    def persona_spoke(self) -> Literal["transfer", "hangup"] | None:
        """Returns 'transfer' when the persona session must speak again (new person picks up)."""
        self.touch()
        if self.persona_hung_up:
            return "hangup"
        if self.transfer_pending:
            self.transfer_pending = False
            return "transfer"
        return None

    # -------------------------------------------------------------- watchdog
    def silent_for(self) -> float:
        return self.clock() - self.last_activity

    def nudge(self) -> str:
        """Caller line after SILENCE_S of silence; after MAX_NUDGES, a goodbye that ends the call."""
        self.touch()
        self.nudges += 1
        _add_flag(self.flags, "silence_nudge")
        if self.nudges > MAX_NUDGES:
            reason = "no_answer" if self.counterparty_count() == 0 else "silence"
            ct = CallerTurn(say=GIVE_UP_TEXT, slots_touched=[], action="hangup", hangup_reason=None)
            self.pending_end = reason
        else:
            ct = CallerTurn(say=NUDGE_TEXT, slots_touched=[], action="continue", hangup_reason=None)
        self.t_caller.add_caller_turn(ct, ts=self._ts())
        self._unheard_caller.append(len(self.t_caller.utterances) - 1)
        return ct.say

    # -------------------------------------------------------------- result
    def raw(self, transport: str = "voice") -> Any:
        from miramace.calls.runner import RawCall

        return RawCall(
            utterances=list(self.t_caller.utterances),
            end_reason=self.end_reason or "aborted",
            transport=transport,
            duration_s=int(round(self.clock() - self.t0)),
            flags=list(self.flags),
        )


# ----------------------------------------------------------------------------------- BrainAgent


def make_brain_agent_cls() -> type:
    """Defined lazily so the class only exists when livekit imports."""
    from livekit.agents import Agent, StopResponse

    class BrainAgent(Agent):
        """Wraps the SAME ports.Brain used by text calls; the session's LLM is never invoked."""

        def __init__(self, ctl: DuplexController, side: Side) -> None:
            super().__init__(instructions=f"miramace {side}: replies come from a Brain via llm_node.")
            self.ctl = ctl
            self.side = side

        async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
            text = (new_message.text_content or "").strip()
            if not text:
                raise StopResponse()
            stop = self.ctl.caller_heard(text) if self.side == "caller" else self.ctl.persona_heard(text)
            if stop:
                raise StopResponse()

        async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any) -> str | None:
            if self.side == "caller":
                return await self.ctl.caller_line()
            return await self.ctl.persona_line()

    return BrainAgent


# ---------------------------------------------------------------------------------- run (live)

TURN_HANDLING: dict[str, Any] = {
    "turn_detection": "vad",  # default is inference.TurnDetector() → LiveKit Cloud
    "endpointing": {"min_delay": 0.7, "max_delay": 3.0},
    "interruption": {"enabled": False},
    "preemptive_generation": {"enabled": False},
}

_TTS_STYLE = {
    "caller": "Calm, clear, professional phone voice. Moderate pace.",
    "persona": "Busy but friendly medical front-desk receptionist on the phone. Natural, brisk pace.",
}


def _make_session(side: Side, task: CallTask) -> Any:
    from livekit.agents import AgentSession, inference
    from livekit.plugins import openai as lk_openai

    voice = (
        os.environ.get("MIRA_TTS_VOICE_CALLER", "sage")
        if side == "caller"
        else os.environ.get("MIRA_TTS_VOICE_PERSONA", "coral")
    )
    stt_prompt = f"Phone call between a care-coordination assistant and {task.org_name}."
    placeholder = os.environ.get("MIRA_TURN_MODEL") or os.environ.get("MIRA_TEXT_MODEL") or "gpt-4.1-mini"
    return AgentSession(
        stt=lk_openai.STT(
            model=os.environ.get("MIRA_STT_MODEL", "gpt-4o-mini-transcribe"), language="en", prompt=stt_prompt
        ),
        llm=lk_openai.LLM(model=placeholder),  # required for replies to run; llm_node overrides it
        tts=lk_openai.TTS(
            model=os.environ.get("MIRA_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=voice,
            instructions=_TTS_STYLE[side],
        ),
        vad=inference.VAD(model="silero", min_silence_duration=0.5),
        turn_handling=TURN_HANDLING,
        user_away_timeout=None,
    )


def _room_options(other_identity: str) -> Any:
    from livekit import rtc
    from livekit.agents.voice import room_io

    return room_io.RoomOptions(
        participant_identity=other_identity,
        participant_kinds=[
            rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
            rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
        ],
        text_input=False,
        close_on_disconnect=False,
    )


def _has_subscribed_audio(room: Any, identity: str) -> bool:
    p = room.remote_participants.get(identity)
    return bool(p) and any(pub.subscribed for pub in p.track_publications.values())


async def _wait_for(pred: Callable[[], bool], timeout: float, step: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()


async def _await_listener(room: str, timeout: float = 60.0) -> bool:
    """Open the listener link and wait (≤timeout) until the hidden listener shows up server-side."""
    from livekit import api

    link = listener_url(room)
    print(f"\n[listen] open in Chrome (allow audio autoplay, mic/cam off):\n  {link}")
    print(f"[listen] or mint a token with the CLI:\n  {listener_cli_hint(room)}\n")
    with contextlib.suppress(Exception):
        webbrowser.open(link)
    url, key, secret = lk_env()
    lkapi = api.LiveKitAPI(url, key, secret)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                res = await lkapi.room.list_participants(api.ListParticipantsRequest(room=room))
                if any(p.identity == LISTENER_IDENTITY for p in res.participants):
                    print("[listen] listener joined")
                    await asyncio.sleep(1.0)
                    return True
            await asyncio.sleep(1.0)
        print(f"[listen] no listener after {int(timeout)}s; continuing")
        return False
    finally:
        await lkapi.aclose()


async def _delete_room(room: str) -> None:
    from livekit import api

    url, key, secret = lk_env()
    lkapi = api.LiveKitAPI(url, key, secret)
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room))
    finally:
        await lkapi.aclose()


async def run_voice_call(
    task: CallTask,
    caller_brain: Any,
    persona_brain: Any,
    persona_spec: PersonaSpec | None,
    *,
    listen: bool = False,
    room: str | None = None,
    timeout_s: float = 240,
) -> Any:
    """One call over local LiveKit with two in-process AgentSessions. Returns calls.runner.RawCall.

    Raises ImportError (no voice extra) or VoiceCallError(reason) — callers fall back to text.
    """
    from livekit import rtc  # ImportError → fallback

    url, _, _ = lk_env()
    room = room or room_name(task)
    pid = persona_identity(task)
    ctl = DuplexController(task, caller_brain, persona_brain, persona_spec)
    agent_cls = make_brain_agent_cls()
    r_caller, r_persona = rtc.Room(), rtc.Room()
    sessions: list[Any] = []
    bg: list[asyncio.Task[Any]] = []
    speaking: dict[str, bool] = {"caller": False, "persona": False}

    try:
        try:
            await asyncio.wait_for(r_caller.connect(url, access_token(CALLER_IDENTITY, room)), 10)
            await asyncio.wait_for(r_persona.connect(url, access_token(pid, room)), 10)
        except Exception as e:
            raise VoiceCallError("connect", f"{type(e).__name__}: {e}") from e
        log.info("voice: connected %s + %s to %s", CALLER_IDENTITY, pid, room)
        if listen:
            await _await_listener(room)

        s_caller = _make_session("caller", task)
        s_persona = _make_session("persona", task)
        sessions += [s_caller, s_persona]

        def on_item(side: Side) -> Callable[[Any], None]:
            def _h(ev: Any) -> None:
                item = ev.item
                role = getattr(item, "role", None)
                text = (getattr(item, "text_content", None) or "").strip()
                if role != "assistant":
                    return
                log.info("voice %s: %s", side, text)
                if side == "caller":
                    ctl.caller_spoke()
                    return
                what = ctl.persona_spoke()
                if what == "transfer":  # the new person picks up and speaks
                    s_persona.generate_reply()
                elif what == "hangup":  # caller may not transcribe the goodbye; end after a grace

                    async def _grace() -> None:
                        await asyncio.sleep(4.0)
                        ctl.end("counterparty_hangup")

                    bg.append(asyncio.create_task(_grace()))

            return _h

        def on_state(side: Side) -> Callable[[Any], None]:
            def _h(ev: Any) -> None:
                speaking[side] = ev.new_state in ("speaking", "thinking")
                if speaking[side]:
                    ctl.touch()

            return _h

        def on_user_state(ev: Any) -> None:
            if ev.new_state == "speaking":
                ctl.touch()

        def on_close(side: Side) -> Callable[[Any], None]:
            def _h(ev: Any) -> None:
                err = getattr(ev, "error", None)
                if err is not None:
                    ctl.fail(f"{side}_session_error:{type(err).__name__}")

            return _h

        for side, s in (("caller", s_caller), ("persona", s_persona)):
            s.on("conversation_item_added", on_item(side))
            s.on("agent_state_changed", on_state(side))
            s.on("user_state_changed", on_user_state)
            s.on("close", on_close(side))

        try:
            await s_caller.start(
                agent_cls(ctl, "caller"),
                room=r_caller,
                room_options=_room_options(pid),
                record=False,
                session_host=False,
            )
            await s_persona.start(
                agent_cls(ctl, "persona"),
                room=r_persona,
                room_options=_room_options(CALLER_IDENTITY),
                record=False,
                session_host=False,
            )
        except Exception as e:
            raise VoiceCallError("session_start", f"{type(e).__name__}: {e}") from e

        # persona speaks first once both sides can hear each other (or after 5 s)
        await _wait_for(
            lambda: (
                _has_subscribed_audio(r_caller, pid) and _has_subscribed_audio(r_persona, CALLER_IDENTITY)
            ),
            5.0,
        )
        await asyncio.sleep(0.5)
        ctl.touch()
        s_persona.generate_reply()

        async def watchdog() -> None:
            while not ctl.ended.is_set():
                await asyncio.sleep(1.0)
                if any(speaking.values()) or ctl.pending_end is not None:
                    continue
                if ctl.silent_for() >= SILENCE_S:
                    text = ctl.nudge()
                    log.info("voice watchdog: %s", text)
                    s_caller.say(text, allow_interruptions=False)

        bg.append(asyncio.create_task(watchdog()))
        try:
            await asyncio.wait_for(ctl.ended.wait(), timeout_s)
        except TimeoutError as e:
            raise VoiceCallError("timeout", f">{int(timeout_s)}s") from e
        if ctl.error:
            raise VoiceCallError(ctl.error)
        return ctl.raw("voice")
    finally:
        for t in bg:
            t.cancel()
        for s in sessions:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(s.aclose(), 8)
        for r in (r_caller, r_persona):
            with contextlib.suppress(Exception):
                await r.disconnect()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(_delete_room(room), 5)


# ------------------------------------------------------------------------------------------ port


def _resolve(factory_or_brain: Any, arg: Any) -> Any:
    if factory_or_brain is None or hasattr(factory_or_brain, "next"):
        return factory_or_brain
    try:
        n = len(inspect.signature(factory_or_brain).parameters)
    except (TypeError, ValueError):
        n = 1
    return factory_or_brain(arg) if n >= 1 else factory_or_brain()


def _reason(e: BaseException) -> str:
    if isinstance(e, VoiceCallError):
        return e.reason
    if isinstance(e, ImportError):
        return "livekit_missing"
    if isinstance(e, TimeoutError):
        return "timeout"
    if isinstance(e, ConnectionError | OSError):
        return "connect"
    return re.sub(r"[^A-Za-z0-9_]+", "_", type(e).__name__)


class VoiceCallerPort:
    """CallerPort: voice over local LiveKit for MIRA_VOICE=1 legs, text otherwise or on failure."""

    def __init__(
        self,
        text_port: Any,
        caller_brain_factory: Any,
        persona_brain_factory: Any,
        persona_lookup: Callable[[str], PersonaSpec | None],
        extractor: Any,
        *,
        listen: bool = False,
        clock: Any = None,
        timeout_s: float = 240,
    ) -> None:
        self.text_port = text_port
        self.caller_brain_factory = caller_brain_factory
        self.persona_brain_factory = persona_brain_factory
        self.persona_lookup = persona_lookup
        self.extractor = extractor
        self.listen = listen
        self.clock = clock
        self.timeout_s = timeout_s
        self.last_error: BaseException | None = None

    async def _voice(self, task: CallTask) -> CallResult:
        from miramace.calls.runner import finalize_call

        caller = _resolve(self.caller_brain_factory, task)
        persona = _resolve(self.persona_brain_factory, task.party_id)
        spec = self.persona_lookup(task.party_id) if self.persona_lookup else getattr(persona, "spec", None)
        ledgers: dict[int, list[LLMCall]] = {}
        for obj in (caller, persona, self.extractor):
            led = getattr(getattr(obj, "llm", None), "ledger", None)
            if isinstance(led, list):
                ledgers[id(led)] = led
        marks = {k: len(v) for k, v in ledgers.items()}
        raw = await run_voice_call(task, caller, persona, spec, listen=self.listen, timeout_s=self.timeout_s)
        n_cp = sum(1 for u in raw.utterances if u.speaker == "counterparty")
        if n_cp < MIN_COUNTERPARTY_UTTERANCES:
            raise VoiceCallError("too_few_utterances", str(n_cp))
        now = self.clock.now() if self.clock is not None else None
        result = await finalize_call(task, raw, self.extractor, now=now)
        usage = [c for k, led in ledgers.items() for c in led[marks[k] :]]
        return result.model_copy(update={"usage": usage, "transport": "voice"})

    async def run_call(self, task: CallTask) -> CallResult:
        if not voice_enabled_for(task):
            return await self.text_port.run_call(task)
        try:
            return await self._voice(task)
        except Exception as e:  # ImportError / connection / timeout / too few utterances / brain error
            self.last_error = e
            flag = f"voice_failed:{_reason(e)}"
            log.warning("voice leg failed (%s); falling back to text", e)
        res = await self.text_port.run_call(task)
        return res.model_copy(update={"transport": "text_fallback", "flags": [*res.flags, flag]})


# ----------------------------------------------------------------------------------------- smoke


def sample_pcp_task(task_id: str = "smoke-pcp-request-order") -> CallTask:
    from miramace.models import SlotSpec, dr_chen, eleanor

    pcp = dr_chen()
    return CallTask(
        task_id=task_id,
        party="pcp",
        party_id=pcp.party_id,
        org_name=pcp.practice,
        phone=pcp.phone,
        intent="request_order",
        purpose=(
            f"Ask {pcp.name}'s office to complete and sign a Standard Written Order for a standard manual "
            "wheelchair (K0001) and fax it with the visit note; learn when it will be signed and by whom."
        ),
        slots=[
            SlotSpec(
                id="request_received", ask="Did you receive our order request?", type="bool", required=True
            ),
            SlotSpec(id="order_status", ask="What is the status of the order?", type="str", required=True),
            SlotSpec(
                id="promised_by", ask="When can Dr. Chen sign and fax it back?", type="date", required=True
            ),
            SlotSpec(id="named_owner", ask="Who will be handling it?", type="str", required=False),
        ],
        phi=eleanor().phi(),
        context={"patient_first_name": "Eleanor"},
        max_turns=14,
    )


def sample_helpful_maria() -> PersonaSpec:
    from miramace.models import Fact, dr_chen

    pcp = dr_chen()
    return PersonaSpec(
        party_id=pcp.party_id,
        role="pcp",
        org_name=pcp.practice,
        answerer_name="Maria",
        behaviour="helpful",
        facts=[
            Fact(
                id="pcp-received",
                slot="request_received",
                value="yes",
                text="Yes, we got your fax this morning.",
                reveal_when="asked whether the order request was received",
            ),
            Fact(
                id="pcp-status",
                slot="order_status",
                value="in_review",
                text="It's in Dr. Chen's signing queue right now.",
                reveal_when="asked about the order status",
            ),
            Fact(
                id="pcp-by",
                slot="promised_by",
                value="Thursday",
                text="She signs on Thursday afternoons, so it'll be faxed back Thursday by five.",
                reveal_when="asked when it will be signed or sent",
            ),
            Fact(
                id="pcp-owner",
                slot="named_owner",
                value="Maria",
                text="I'm handling it myself, it's Maria.",
                reveal_when="asked who is handling it",
            ),
        ],
    )


def _print_result(res: CallResult) -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()
    con.rule(f"[bold]{res.task_id}[/] — transport={res.transport} outcome={res.outcome} {res.duration_s}s")
    tab = Table(show_lines=False, expand=True)
    for col in ("#", "t(s)", "who", "text (caller side)", "heard_text / event"):
        tab.add_column(col, overflow="fold")
    for u in res.transcript:
        extra = " ".join(x for x in (u.heard_text or "", f"[{u.event}]" if u.event else "") if x)
        tab.add_row(str(u.turn), f"{u.ts:.1f}", u.name, u.text, extra)
    con.print(tab)
    for c in res.claims:
        con.print(f"  claim {c.field}: {c.status} {c.value!r}  « {c.evidence_quote} »")
    con.print(f"  flags: {res.flags}")


async def smoke(listen: bool) -> CallResult:
    """`mira call pcp --voice --listen`: Eleanor request_order to a helpful Maria, live OpenAI brains."""
    from miramace.calls.brains import LLMCallerBrain, LLMPersonaBrain
    from miramace.calls.runner import TextCallerPort
    from miramace.extract.extractor import Extractor
    from miramace.llm.port import make_llm

    os.environ.setdefault("MIRA_VOICE", "1")
    os.environ.setdefault("MIRA_VOICE_LEGS", "pcp")
    llm = make_llm("live")
    spec = sample_helpful_maria()
    extractor = Extractor(llm)

    def caller_factory(task: CallTask) -> Any:
        return LLMCallerBrain(llm)

    def persona_factory(party_id: str) -> Any:
        return LLMPersonaBrain(llm, spec)

    text_port = TextCallerPort(caller_factory, persona_factory, extractor)
    port = VoiceCallerPort(
        text_port, caller_factory, persona_factory, lambda pid: spec, extractor, listen=listen
    )
    res = await port.run_call(sample_pcp_task())
    if port.last_error is not None:
        print(f"[voice] fell back to text: {port.last_error!r}")
    _print_result(res)
    return res


if __name__ == "__main__":  # python -m miramace.voice.transport [--listen]
    import sys

    logging.basicConfig(level=logging.INFO)
    asyncio.run(smoke("--listen" in sys.argv))
