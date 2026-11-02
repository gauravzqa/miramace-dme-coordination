"""B3 AI calls — Transcript, phases, disclosure, run_text_call, finalize_call, TextCallerPort.

Code owns the call's control flow; brains only produce the next line:
- the caller's first line, and its first line after any counterparty `transfer`, is the DISCLOSURE template;
- phase (opening → verify_party → verified → readback → closing) is computed by `phase_from`;
- `guard_caller_turn` blocks PHI before verification and unauthorized commitments for every brain.
Shared with the voice transport (B5): Transcript, phase_from, needs_disclosure, guard_caller_turn,
finalize_call.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from miramace.agents.prompts import DISCLOSURE, PHI_PHASES, contains_phi
from miramace.agents.schemas import CallerTurn, PersonaTurn
from miramace.clock import is_business, next_business_time
from miramace.extract.extractor import looks_like_injection, normalize
from miramace.models import (
    CHICAGO,
    CallOutcome,
    CallResult,
    CallTask,
    CallTransport,
    Claim,
    LLMCall,
    PersonaSpec,
    Utterance,
)
from miramace.ports import Brain, CallPhase, TranscriptView

SECONDS_PER_UTTERANCE = 6.0
CALLER_NAME = "Mira Mace AI"

_COMMITMENT_RE = re.compile(
    r"\b(she agrees|he agrees|she'?ll pay|she will pay|we accept|schedule it for|go ahead and)\b", re.I
)
_CALLBACK_RE = re.compile(r"\b(call (us |me )?back|callback)\b")
_CALLBACK_HOUR_RE = re.compile(r"after\s+(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)


# ------------------------------------------------------------------------------------- transcript


def touched_slots(utterances: list[Utterance], slots_by_turn: dict[int, list[str]]) -> set[str]:
    """Slots asked by caller lines, except when the next counterparty line was a transfer/hold/injection
    (the question then never really reached a person and must be asked again)."""
    out: set[str] = set()
    for i, u in enumerate(utterances):
        if u.speaker != "caller" or not slots_by_turn.get(u.turn):
            continue
        nxt = next((v for v in utterances[i + 1 :] if v.speaker == "counterparty"), None)
        if nxt is not None and (nxt.event in ("transfer", "hold") or looks_like_injection(nxt.text)):
            continue
        out.update(slots_by_turn[u.turn])
    return out


class Transcript:
    """Append-only call transcript. `turn` = 0-based index; ts = simulated seconds (6 s per utterance)."""

    def __init__(self) -> None:
        self.utterances: list[Utterance] = []
        self.slots_by_turn: dict[int, list[str]] = {}

    def add(
        self,
        speaker: Literal["caller", "counterparty", "system"],
        name: str,
        text: str,
        event: str | None = None,
        heard_text: str | None = None,
        *,
        slots_touched: list[str] | tuple[str, ...] = (),
        ts: float | None = None,
    ) -> Utterance:
        turn = len(self.utterances)
        u = Utterance(
            turn=turn,
            speaker=speaker,
            name=name,
            text=text,
            ts=float(turn * SECONDS_PER_UTTERANCE) if ts is None else float(ts),
            heard_text=heard_text,
            event=event,
        )
        self.utterances.append(u)
        if speaker == "caller" and slots_touched:
            self.slots_by_turn[turn] = list(slots_touched)
        return u

    def add_caller_turn(
        self,
        turn: CallerTurn,
        name: str = CALLER_NAME,
        heard_text: str | None = None,
        ts: float | None = None,
    ) -> Utterance:
        return self.add(
            "caller",
            name,
            turn.say,
            event="hangup" if turn.action == "hangup" else None,
            heard_text=heard_text,
            slots_touched=turn.slots_touched,
            ts=ts,
        )

    def add_persona_turn(
        self, turn: PersonaTurn, name: str, heard_text: str | None = None, ts: float | None = None
    ) -> Utterance:
        ev = None if turn.event == "none" else turn.event
        return self.add("counterparty", name, turn.say, event=ev, heard_text=heard_text, ts=ts)

    def counterparty_text(self) -> str:
        return "\n".join(u.text for u in self.utterances if u.speaker == "counterparty")

    def touched(self) -> set[str]:
        return touched_slots(self.utterances, self.slots_by_turn)

    def view(
        self,
        task: CallTask,
        phase: str,
        me: Literal["caller", "counterparty"],
        persona: PersonaSpec | None = None,
    ) -> TranscriptView:
        return TranscriptView(
            task=task,
            phase=phase,
            utterances=list(self.utterances),
            me=me,
            persona=persona if me == "counterparty" else None,
        )


def _last_transfer_index(utterances: list[Utterance]) -> int:
    idx = -1
    for i, u in enumerate(utterances):
        if u.speaker == "counterparty" and u.event == "transfer":
            idx = i
    return idx


def is_verified(utterances: list[Utterance], org_name: str) -> bool:
    """A counterparty utterance after the last transfer contains the normalized org name."""
    org = normalize(org_name)
    start = _last_transfer_index(utterances) + 1
    return bool(org) and any(
        u.speaker == "counterparty" and org in normalize(u.text) for u in utterances[start:]
    )


def needs_disclosure(t: Transcript | list[Utterance]) -> bool:
    """True before the caller's first line and right after a counterparty transfer."""
    utts = t.utterances if isinstance(t, Transcript) else t
    callers = [i for i, u in enumerate(utts) if u.speaker == "caller"]
    if not callers:
        return True
    return _last_transfer_index(utts) > callers[-1]


def phase_from(transcript: Transcript, task: CallTask) -> CallPhase:
    utts = transcript.utterances
    caller_idx = [i for i, u in enumerate(utts) if u.speaker == "caller"]
    if not caller_idx:
        return "opening"
    if not is_verified(utts, task.org_name):
        return "verify_party"
    required = {s.id for s in task.slots if s.required}
    done_at: int | None = None
    for i in caller_idx:
        if required <= touched_slots(utts[: i + 1], transcript.slots_by_turn):
            done_at = i
            break
    if done_at is None:
        return "verified"
    readback_idx = next((i for i in caller_idx if i > done_at), None)
    if readback_idx is None:
        return "readback"
    if any(u.speaker == "counterparty" for u in utts[readback_idx + 1 :]):
        return "closing"
    return "readback"


def guard_caller_turn(task: CallTask, phase: str, turn: CallerTurn) -> tuple[CallerTurn, list[str]]:
    """Code-level guardrails on any caller line (LLM or fake): PHI gate, no commitments, valid slot ids."""
    flags: list[str] = []
    valid = {s.id for s in task.slots}
    slots = [s for s in turn.slots_touched if s in valid]
    say = turn.say.strip()
    if phase not in PHI_PHASES and contains_phi(say, task.phi):
        flags.append("phi_blocked")
        say, slots = f"Before we go further, can you confirm I've reached {task.org_name}?", []
    if _COMMITMENT_RE.search(say):
        flags.append("commitment_blocked")
        say, slots = (
            "I can't agree to anything on the patient's behalf, but I'll pass that to the care team.",
            [],
        )
    if not say:
        say = "Sorry, could you repeat that?"
    if say == turn.say and slots == turn.slots_touched:
        return turn, flags
    return turn.model_copy(update={"say": say, "slots_touched": slots}), flags


# ------------------------------------------------------------------------------------------ run


class RawCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    utterances: list[Utterance]
    end_reason: str
    transport: CallTransport
    duration_s: int
    flags: list[str]


def _add_flag(flags: list[str], f: str) -> None:
    if f not in flags:
        flags.append(f)


async def run_text_call(task: CallTask, caller: Brain, persona: Brain) -> RawCall:
    """Persona answers first; caller disclosure is templated; stop on hangup or after task.max_turns
    caller turns."""
    t = Transcript()
    spec: PersonaSpec | None = getattr(persona, "spec", None)
    cp_name = spec.answerer_name if spec else task.org_name
    if spec is not None and spec.behaviour == "ivr":
        cp_name = "Phone menu"
    flags: list[str] = []

    async def persona_turn() -> PersonaTurn:
        nonlocal cp_name
        pv = t.view(task, phase_from(t, task), "counterparty", spec)
        pt = await persona.next(pv)
        assert isinstance(pt, PersonaTurn), "persona brain must return PersonaTurn"
        t.add_persona_turn(pt, cp_name)
        if looks_like_injection(pt.say):
            _add_flag(flags, "injection_suspected")
        if pt.event == "transfer":
            _add_flag(flags, "transferred")
            cp_name = pt.transfer_to or "Transferred party"
        return pt

    end_reason: str | None = None
    pt = await persona_turn()
    if pt.event == "hangup":
        end_reason = "counterparty_hangup"
    elif pt.event == "transfer":
        pt = await persona_turn()
        if pt.event == "hangup":
            end_reason = "counterparty_hangup"
    turns = 0
    while end_reason is None:
        if turns >= task.max_turns:
            end_reason = "max_turns"
            _add_flag(flags, "max_turns")
            break
        turns += 1
        phase = phase_from(t, task)
        if needs_disclosure(t):
            ct = CallerTurn(say=DISCLOSURE, slots_touched=[], action="continue", hangup_reason=None)
        else:
            out = await caller.next(t.view(task, phase, "caller"))
            assert isinstance(out, CallerTurn), "caller brain must return CallerTurn"
            ct, gflags = guard_caller_turn(task, phase, out)
            for f in gflags:
                _add_flag(flags, f)
        t.add_caller_turn(ct)
        if ct.action == "hangup":
            end_reason = ct.hangup_reason or "complete"
            break
        pt = await persona_turn()
        if pt.event == "transfer":  # the new person picks up
            pt = await persona_turn()
        if pt.event == "hangup":
            end_reason = "counterparty_hangup"
    return RawCall(
        utterances=t.utterances,
        end_reason=end_reason,
        transport="text",
        duration_s=int(len(t.utterances) * SECONDS_PER_UTTERANCE),
        flags=flags,
    )


# ------------------------------------------------------------------------------------- finalize

_REASON_OUTCOME: dict[str, CallOutcome] = {
    "complete": "COMPLETED",
    "refused": "REFUSED",
    "wrong_number": "WRONG_NUMBER",
    "voicemail": "VOICEMAIL",
    "callback_later": "CALLBACK_REQUESTED",
    "identity_failed": "FAILED",
    "injection": "FAILED",
    "no_answer": "NO_ANSWER",
}


def _callback_hour(text: str) -> tuple[int, int] | None:
    m = _CALLBACK_HOUR_RE.search(text)
    if not m:
        return None
    h, mins = int(m.group(1)), int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower().replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    elif not ampm and 1 <= h < 8:
        h += 12
    if not 0 <= h <= 23 or not 0 <= mins <= 59:
        return None
    return h, mins


def callback_time(text: str, now: datetime) -> datetime | None:
    hm = _callback_hour(text)
    if hm is None:
        return None
    now = now.astimezone(CHICAGO)
    cand = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=1)
    for _ in range(7):  # same clock time on the next business day
        if is_business(cand):
            return cand
        cand += timedelta(days=1)
    return next_business_time(cand)


def _required_stated(task: CallTask, claims: list[Claim]) -> bool:
    stated = {c.field for c in claims if c.status == "stated"}
    return all(s.id in stated for s in task.slots if s.required)


async def finalize_call(
    task: CallTask,
    raw: RawCall,
    extractor: Any,
    *,
    now: datetime | None = None,
    usage: list[LLMCall] | None = None,
) -> CallResult:
    """Shared by text and voice: extraction (two-pass + code checks) and end_reason → CallOutcome."""
    flags = list(raw.flags)
    claims: list[Claim] = []
    if any(u.speaker == "counterparty" for u in raw.utterances):
        try:
            claims, xflags = await extractor.extract_verified(task, raw.utterances)
            flags.extend(f for f in xflags if f not in flags)
        except Exception as e:  # extraction must never crash the call pipeline
            _add_flag(flags, f"extraction_failed:{type(e).__name__}")
    cp = [u for u in raw.utterances if u.speaker == "counterparty"]
    last_cp = cp[-1].text if cp else ""
    reason = raw.end_reason
    outcome: CallOutcome
    if reason in _REASON_OUTCOME:
        outcome = _REASON_OUTCOME[reason]
    elif reason == "counterparty_hangup" and _CALLBACK_RE.search(normalize(last_cp)):
        outcome = "CALLBACK_REQUESTED"
    elif reason in ("counterparty_hangup", "max_turns"):
        outcome = "COMPLETED" if claims and _required_stated(task, claims) else "FAILED"
        if reason == "counterparty_hangup":
            _add_flag(flags, "counterparty_hangup")
    else:
        outcome = "FAILED"
        _add_flag(flags, f"end_reason:{reason}")
    callback_after: datetime | None = None
    if outcome == "CALLBACK_REQUESTED":
        text = next((u.text for u in reversed(cp) if _CALLBACK_HOUR_RE.search(u.text)), last_cp)
        hm = _callback_hour(text)
        if hm is not None:
            _add_flag(flags, f"callback_after_hour:{hm[0]}")
            if now is not None:
                callback_after = callback_time(text, now)
    return CallResult(
        task_id=task.task_id,
        outcome=outcome,
        transcript=raw.utterances,
        claims=claims,
        flags=flags,
        duration_s=raw.duration_s,
        callback_after=callback_after,
        transport=raw.transport,
        usage=list(usage or []),
    )


# ------------------------------------------------------------------------------------- port


def _resolve(factory_or_brain: Any, arg: Any) -> Any:
    if hasattr(factory_or_brain, "next"):
        return factory_or_brain
    try:
        n = len(inspect.signature(factory_or_brain).parameters)
    except (TypeError, ValueError):
        n = 1
    return factory_or_brain(arg) if n >= 1 else factory_or_brain()


class TextCallerPort:
    """CallerPort over text: run_text_call + finalize_call.

    caller_brain_factory: a Brain, a zero-arg factory, or factory(task) -> Brain.
    persona_brain_factory: a Brain or factory(party_id) -> Brain (persona spec lookup is the factory's job).
    clock: optional object with .now() used to resolve callback_after.
    """

    def __init__(
        self, caller_brain_factory: Any, persona_brain_factory: Any, extractor: Any, *, clock: Any = None
    ) -> None:
        self.caller_brain_factory = caller_brain_factory
        self.persona_brain_factory = persona_brain_factory
        self.extractor = extractor
        self.clock = clock

    async def run_call(self, task: CallTask) -> CallResult:
        caller = _resolve(self.caller_brain_factory, task)
        persona = _resolve(self.persona_brain_factory, task.party_id)
        ledgers: dict[int, list[LLMCall]] = {}
        for obj in (caller, persona, self.extractor):
            led = getattr(getattr(obj, "llm", None), "ledger", None)
            if isinstance(led, list):
                ledgers[id(led)] = led
        marks = {k: len(v) for k, v in ledgers.items()}
        raw = await run_text_call(task, caller, persona)
        now = self.clock.now() if self.clock is not None else None
        result = await finalize_call(task, raw, self.extractor, now=now)
        usage = [c for k, led in ledgers.items() for c in led[marks[k] :]]
        return result.model_copy(update={"usage": usage}) if usage else result
