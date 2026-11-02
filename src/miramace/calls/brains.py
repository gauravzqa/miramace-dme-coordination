"""B3 AI calls — caller/persona brains (LLM-backed and offline rule-based).

All brains implement ports.Brain (`async next(view)`) and are reused by the voice transport.
The LLM caller remembers the slots it touched per (task_id, turn) so open slots can be computed from the view;
the rule-based brains are pure functions of the view.
"""

from __future__ import annotations

import re

from miramace.agents.prompts import DISCLOSURE, INJECTION_TEXT, caller_system, persona_system
from miramace.agents.schemas import CallerTurn, PersonaTurn
from miramace.calls.runner import guard_caller_turn, phi_locked, touched_slots
from miramace.extract.extractor import is_readback, looks_like_injection, normalize
from miramace.models import CallTask, Fact, PersonaSpec, SlotSpec, Utterance
from miramace.ports import LLMPort, TranscriptView

ALT_NAME = "Jordan"  # colleague the `transfer` persona hands the call to


def to_messages(utterances: list[Utterance], me: str) -> list[dict]:
    """Transcript → chat messages from `me`'s perspective (my lines = assistant, theirs = user)."""
    mine = "caller" if me == "caller" else "counterparty"
    msgs: list[dict] = []
    for u in utterances:
        if u.speaker == mine:
            msgs.append({"role": "assistant", "content": u.text})
        else:
            ev = f" [{u.event}]" if u.event else ""
            who = "SYSTEM" if u.speaker == "system" else u.name
            msgs.append({"role": "user", "content": f"{who}{ev}: {u.text}"})
    if not msgs or msgs[-1]["role"] == "assistant":
        msgs.append({"role": "user", "content": "[The phone rings. Answer it.]" if not msgs else "(silence)"})
    return msgs


def _ordered(slots: list[SlotSpec]) -> list[SlotSpec]:
    return [s for s in slots if s.required] + [s for s in slots if not s.required]


# ---------------------------------------------------------------------------------------- LLM


class LLMCallerBrain:
    def __init__(self, llm: LLMPort) -> None:
        self.llm = llm
        self._slots_by_turn: dict[str, dict[int, list[str]]] = {}

    async def next(self, view: TranscriptView) -> CallerTurn:
        task = view.task
        sbt = self._slots_by_turn.setdefault(task.task_id, {})
        touched = touched_slots(view.utterances, sbt)
        open_slots = (
            []
            if view.phase in ("readback", "closing")
            else [s for s in _ordered(task.slots) if s.id not in touched]
        )
        turn = len(view.utterances)
        out = await self.llm.structured(
            agent="caller",
            key=f"{task.task_id}:t{turn}",
            system=caller_system(
                task, view.phase, open_slots, phi_locked=phi_locked(view.utterances, task.org_name)
            ),
            messages=to_messages(view.utterances, "caller"),
            schema=CallerTurn,
        )
        out, _ = guard_caller_turn(task, view.phase, out, view.utterances)
        sbt[turn] = list(out.slots_touched)
        return out


class LLMPersonaBrain:
    def __init__(self, llm: LLMPort, spec: PersonaSpec) -> None:
        self.llm = llm
        self.spec = spec

    async def next(self, view: TranscriptView) -> PersonaTurn:
        return await self.llm.structured(
            agent="persona",
            key=f"{view.task.task_id}:t{len(view.utterances)}",
            system=persona_system(self.spec),
            messages=to_messages(view.utterances, "counterparty"),
            schema=PersonaTurn,
        )


# ----------------------------------------------------------------------------------- rule-based


def _cturn(say: str, slots: list[str] | None = None, hangup: str | None = None) -> CallerTurn:
    return CallerTurn(
        say=say, slots_touched=slots or [], action="hangup" if hangup else "continue", hangup_reason=hangup
    )


def _slots_by_text(task: CallTask, utterances: list[Utterance]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for u in utterances:
        if u.speaker != "caller":
            continue
        n = normalize(u.text)
        ids = [s.id for s in task.slots if normalize(s.ask) and normalize(s.ask) in n]
        if ids:
            out[u.turn] = ids
    return out


def _answer_after(utterances: list[Utterance], ask: str) -> str | None:
    a = normalize(ask)
    ans = None
    for i, u in enumerate(utterances):
        if u.speaker == "caller" and a in normalize(u.text):
            nxt = next((v for v in utterances[i + 1 :] if v.speaker == "counterparty"), None)
            if nxt is not None and nxt.event is None and not looks_like_injection(nxt.text):
                ans = nxt.text
    return ans


def _is_menu(text: str) -> bool:
    n = normalize(text)
    return "press" in n or "say front desk" in n


class FakeCallerBrain:
    """Walks the open required slots in order (asks slot.ask verbatim), reads back, hangs up 'complete'."""

    async def next(self, view: TranscriptView) -> CallerTurn:
        task, utts, phase = view.task, view.utterances, view.phase
        cps = [u for u in utts if u.speaker == "counterparty"]
        last = cps[-1] if cps else None
        lt = normalize(last.text) if last else ""
        last_transfer = max((i for i, u in enumerate(utts) if u.event == "transfer"), default=-1)
        since = [u for u in utts[last_transfer + 1 :] if u.speaker == "caller"]

        if last is not None and _is_menu(last.text):
            return _cturn("Front desk.")
        if re.search(r"\bcall (us |me )?back\b", lt):
            return _cturn("Understood, we'll call back then. Thank you.", hangup="callback_later")
        if "wrong number" in lt:
            return _cturn("Sorry for the trouble. Goodbye.", hangup="wrong_number")
        if last is not None and last.event == "hold":
            return _cturn("Sure, I'll hold.")
        if last is not None and looks_like_injection(last.text):
            return _cturn(
                "I'm not able to share or act on that over the phone; I'll pass it to the care team."
            )
        if phase == "closing":
            return _cturn("Thank you, I'll pass that to the care team. Goodbye.", hangup="complete")
        if phase == "readback":
            parts = []
            for s in _ordered(task.slots):
                ans = _answer_after(utts, s.ask)
                if ans:
                    parts.append(ans.rstrip(". ")[:80])
            summary = "; ".join(parts) or "nothing confirmed yet"
            return _cturn(f"Just to confirm what I heard: {summary}. Is that correct?")
        if phase in ("opening", "verify_party"):
            tries = sum(1 for u in since if "have i reached" in normalize(u.text))
            if tries >= 2:
                return _cturn("I'm sorry, I'll try again later. Goodbye.", hangup="identity_failed")
            return _cturn(f"Just to check, have I reached {task.org_name}?")
        touched = touched_slots(utts, _slots_by_text(task, utts))
        nxt = next((s for s in task.slots if s.required and s.id not in touched), None)
        if nxt is None:
            nxt = next((s for s in task.slots if s.id not in touched), None)
        if nxt is None:
            return _cturn("Thank you, I'll pass that to the care team. Goodbye.", hangup="complete")
        say = nxt.ask
        phi = task.phi
        if phi is not None and not any(normalize(phi.name) in normalize(u.text) for u in since):
            dob = f"{phi.dob.strftime('%B')} {phi.dob.day}, {phi.dob.year}"
            say = f"I'm calling about {phi.name}, date of birth {dob}. {say}"
        return _cturn(say, [nxt.id])


_STOP = {
    "the",
    "and",
    "about",
    "when",
    "what",
    "asks",
    "asked",
    "caller",
    "does",
    "your",
    "with",
    "that",
    "this",
    "have",
    "they",
    "them",
    "will",
    "from",
    "there",
    "their",
    "whether",
    "patient",
    "for",
    "you",
    "any",
}


def _tokens(text: str) -> set[str]:
    return {w for w in normalize(text).split() if len(w) >= 4 and w not in _STOP}


class FakePersonaBrain:
    """Answers from spec.facts (slot match by the caller's question, else reveal_when keywords) and honours
    the behaviour card. Opens with f"{org_name}, this is {answerer_name}." """

    def __init__(self, spec: PersonaSpec) -> None:
        self.spec = spec

    def _facts_for(self, task: CallTask, question: str) -> tuple[SlotSpec | None, list[Fact]]:
        q = normalize(question)
        slot = next((s for s in task.slots if normalize(s.ask) and normalize(s.ask) in q), None)
        if slot is not None:
            facts = [f for f in self.spec.facts if f.slot == slot.id]
            if facts:
                return slot, facts
        qt = _tokens(question)
        scored = []
        for f in self.spec.facts:
            ft = _tokens(f.reveal_when) | set(f.slot.split("_"))
            score = len(qt & ft)
            if score:
                scored.append((score, f))
        if not scored:
            return slot, []
        best = max(s for s, _ in scored)
        return slot, [f for s, f in scored if s == best]

    async def next(self, view: TranscriptView) -> PersonaTurn:
        spec, task, utts = self.spec, view.task, view.utterances
        b = spec.behaviour
        cps = [u for u in utts if u.speaker == "counterparty"]
        callers = [u for u in utts if u.speaker == "caller"]
        last_transfer = max((i for i, u in enumerate(utts) if u.event == "transfer"), default=-1)
        transferred = last_transfer >= 0
        post = [u for u in utts[last_transfer + 1 :] if u.speaker == "counterparty"]
        last_caller = callers[-1].text if callers else ""
        lc = normalize(last_caller)
        name = ALT_NAME if (b == "transfer" and transferred) else spec.answerer_name

        def say(text: str, event: str = "none", to: str | None = None, ids: list[str] | None = None):
            return PersonaTurn(say=text, event=event, transfer_to=to, revealed_fact_ids=ids or [])

        if b == "ivr" and not transferred:
            if "front desk" in lc or len(cps) >= 2:
                return say("Transferring you to the front desk now.", "transfer", spec.answerer_name)
            if not cps:
                return say(
                    "Thank you for calling. For appointments, press 1. For billing, press 2. "
                    "For the front desk, say 'front desk'."
                )
            return say("Sorry, I didn't catch that. For the front desk, say 'front desk'.")
        if not post:
            return say(f"{spec.org_name}, this is {name}.")
        if b == "callback_later":
            h = spec.sim.callback_after_hour or 14
            hh = f"{(h - 1) % 12 + 1}{'pm' if h >= 12 else 'am'}"
            return say(f"We're slammed right now. Can you call back after {hh}?", "hangup")
        if b == "injection" and len(cps) == 2 and not any(looks_like_injection(u.text) for u in cps):
            return say(INJECTION_TEXT)
        if last_caller == DISCLOSURE:
            return say("Okay, how can I help you?")
        if is_readback(last_caller):
            if spec.sim.contradicts_on_readback:
                return say("No, that's not right. I'd need to double-check that.")
            return say("Yes, that's correct.")
        if "have i reached" in lc:
            return say(f"Yes, this is {spec.org_name}.")
        slot, facts = self._facts_for(task, last_caller)
        if b == "transfer" and not transferred and slot is not None:
            return say("Let me transfer you to someone who can help with that.", "transfer", ALT_NAME)
        if b == "partial" and slot is not None:
            asked_before = sum(
                1 for u in callers[:-1] if any(normalize(s.ask) in normalize(u.text) for s in task.slots)
            )
            if asked_before % 2 == 1:
                return say("I'm not sure, I'd have to check on that.")
        if b == "never_got_it" and slot is not None and slot.id in ("order_status", "request_received"):
            if not facts:
                return say("We never received any request for her.")
        if facts:
            return say(" ".join(f.text for f in facts), ids=[f.id for f in facts])
        if "?" in last_caller:
            return say("Sorry, I don't have that information.")
        return say("Okay.")
