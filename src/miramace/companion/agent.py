"""B4 Patient side — LLM companion implementing CompanionPort.

respond: LLM (schema CompanionOut) -> grounding check -> regenerate once with feedback -> else safe template
+ flag grounding_hold; then keyword safety net (distress / "a person please" -> needs_human, always).
compose: LLM wording of a code-chosen update -> grounding -> regenerate once -> else notify.render_template.
The companion never changes state: it returns typed Intents for intents.IntentAPI to validate.
"""

from __future__ import annotations

import re

from miramace.agents.schemas import CompanionOut, IntentLLM
from miramace.grounding import grounding_check
from miramace.models import CompanionReply, ComposeRequest, Intent, PatientProfile, Snapshot, Utterance
from miramace.notify import render_template, safe_reply
from miramace.ports import LLMPort
from miramace.snapshot import snapshot_text

COMPANION_SYSTEM = """You are Mira Mace's assistant for Eleanor, a 72-year-old woman getting a standard \
manual wheelchair through Original Medicare Part B. You are warm, patient and brief (2-4 short sentences, \
plain words, no jargon). A nurse is always available to her.

The ONLY facts you may state are in the CASE SNAPSHOT below. Do not mention any date, day, dollar amount, \
percentage, company or person that is not written in the snapshot. If she asks something the snapshot does \
not answer, say you'll find out and let her know. Never promise dates or costs beyond the snapshot. Never \
agree, consent, pay or schedule anything on her behalf; you only record what SHE tells you.

Report her intents (empty list if none):
- confirm_address: she gives her home address (put the full address in `address`, any home details in `note`)
- give_consent: she says it's OK for us to arrange the wheelchair for her
- ask_status: she asks where things stand
- cost_concern: she is worried about cost or can't afford it
- cost_accepted: she says the explained cost works for her
- confirm_delivery_slot: she accepts the proposed delivery time
- reschedule: she needs to move a scheduled delivery (`preferred_time` if she gives one)
- needs_human: distress, pain, a fall, fear, a clinical question, confusion, or she asks for a person/nurse \
(`note`: distress | clinical | confusion | human)
- delivery_confirmed: she confirms the wheelchair arrived and works
If she is worried, scared, hurt, or asks for a person, ALWAYS include needs_human (or cost_concern for cost) \
and tell her a nurse will contact her.

{snapshot}"""

COMPOSE_SYSTEM = """You are Mira Mace's assistant writing a short message to Eleanor (72) about her \
wheelchair. Warm, plain words, 2-4 short sentences. PURPOSE of this message: {purpose}. {purpose_hint}
Use ONLY facts written in the CASE SNAPSHOT below: no other dates, days, dollar amounts, percentages, \
companies or people. Never agree to anything for her. Return the message in `reply` and an empty \
`intents` list.
{profile}
{snapshot}"""

PURPOSE_HINTS = {
    "welcome_intake": "Introduce yourself, say a nurse is always available, and ask for her home address "
    "(with ZIP) and her OK to arrange the wheelchair for her.",
    "cost_disclosure": "Explain her cost exactly as the snapshot's Cost line says, then ask if that works "
    "for her and invite any worries.",
    "delivery_slot": "Tell her the proposed delivery time from the snapshot and ask if it works.",
    "delivered_checkin": "Ask whether the wheelchair arrived, is working, and fits through her doorways.",
    "status_update": "Tell her briefly what changed, using the snapshot headline.",
}

_DISTRESS_RE = re.compile(
    r"\b(pain|painful|hurt|hurts|hurting|fell|fall|falls|falling|fallen|scared|afraid|frightened|dizzy|"
    r"chest|bleeding|bleed|blood|emergency|911|ambulance|unconscious|passed out|pass out|fainted|faint|"
    r"can'?t breathe|cannot breathe|can not breathe|short of breath|trouble breathing|hard to breathe|"
    r"confused|confusing|stroke|help me)\b",
    re.IGNORECASE,
)
_HUMAN_RE = re.compile(
    r"\bnurse\b|talk to a person|real person|speak to someone|talk to someone|speak to a person|a human|"
    r"talk to a human|speak to a human|someone real",
    re.IGNORECASE,
)

HANDOFF_SENTENCE = "I'm letting a nurse know right now, and they'll reach out to you."


def safety_net_kind(text: str) -> str | None:
    """'distress' / 'human' if the keyword safety net fires on the patient's text, else None."""
    text = text.replace("\u2019", "'")
    if _DISTRESS_RE.search(text):
        return "distress"
    if _HUMAN_RE.search(text):
        return "human"
    return None


def to_intent(i: IntentLLM, utterance: str, conversation_id: str) -> Intent:
    return Intent(
        kind=i.kind,
        address=i.address,
        preferred_time=i.preferred_time,
        note=i.note,
        utterance=utterance,
        conversation_id=conversation_id,
    )


def _history_messages(history: list[Utterance]) -> list[dict]:
    """Companion lines are speaker 'caller' (our side); Eleanor's are 'counterparty'; 'system' is dropped."""
    out: list[dict] = []
    for u in history:
        if u.speaker == "system":
            continue
        out.append({"role": "assistant" if u.speaker == "caller" else "user", "content": u.text})
    return out


class LLMCompanion:
    """CompanionPort backed by an LLMPort."""

    def __init__(
        self, llm: LLMPort, conversation_id: str | None = None, profile: PatientProfile | None = None
    ) -> None:
        self.llm = llm
        self.conversation_id = conversation_id
        self.profile = profile

    async def _ask(self, key: str, system: str, messages: list[dict]) -> CompanionOut | None:
        try:
            return await self.llm.structured(
                agent="companion", key=key, system=system, messages=messages, schema=CompanionOut
            )
        except Exception:
            return None

    async def _grounded(
        self, key: str, system: str, messages: list[dict], snapshot: Snapshot, profile: PatientProfile | None
    ) -> tuple[CompanionOut | None, list[str], list[str]]:
        """Generate, check, regenerate once. Returns (out or None if held, flags, last violations)."""
        flags: list[str] = []
        out = await self._ask(key, system, messages)
        if out is None:
            return None, ["llm_unavailable"], []
        violations = grounding_check(out.reply, snapshot, profile)
        if not violations:
            return out, flags, []
        flags.append("grounding_regenerated")
        feedback = (
            f"Your reply mentioned {'; '.join(violations)}, which is not in the CASE SNAPSHOT. Rewrite it "
            "using only snapshot facts; if you don't know, say you'll find out."
        )
        retry_msgs = [
            *messages,
            {"role": "assistant", "content": out.reply},
            {"role": "user", "content": feedback},
        ]
        out2 = await self._ask(f"{key}_retry", system, retry_msgs)
        if out2 is not None:
            v2 = grounding_check(out2.reply, snapshot, profile)
            if not v2:
                return out2, flags, []
            violations = v2
            # keep intents from the retry if it produced any
            out = out2 if out2.intents else out
        flags.append("grounding_hold")
        return CompanionOut(reply="", intents=out.intents), flags, violations

    async def respond(self, snapshot: Snapshot, history: list[Utterance], text: str) -> CompanionReply:
        conv = self.conversation_id or f"conv-{snapshot.case_id}"
        system = COMPANION_SYSTEM.format(snapshot=snapshot_text(snapshot))
        messages = [*_history_messages(history), {"role": "user", "content": text}]
        out, flags, _ = await self._grounded("respond", system, messages, snapshot, self.profile)

        reply = out.reply if out is not None and out.reply else safe_reply(snapshot)
        intents = [to_intent(i, text, conv) for i in (out.intents if out is not None else [])]

        # Keyword safety net: distress / "a person please" ALWAYS yields needs_human, whatever else the LLM
        # reported (a fall + a cost worry keeps both intents).
        kind = safety_net_kind(text)
        if kind and not any(i.kind == "needs_human" for i in intents):
            intents.append(Intent(kind="needs_human", note=kind, utterance=text, conversation_id=conv))
            flags.append("safety_net_needs_human")
        if any(i.kind in ("needs_human", "cost_concern") for i in intents) and "nurse" not in reply.lower():
            reply = f"{reply} {HANDOFF_SENTENCE}"

        ok = not grounding_check(reply, snapshot, self.profile)
        return CompanionReply(reply=reply, intents=intents, grounding_ok=ok, flags=flags)

    async def compose(self, req: ComposeRequest) -> str:
        profile = req.profile
        prof_txt = ""
        if profile.caregiver_name:
            prof_txt = f"Her caregiver is {profile.caregiver_name}."
        system = COMPOSE_SYSTEM.format(
            purpose=req.purpose,
            purpose_hint=PURPOSE_HINTS.get(req.purpose, ""),
            profile=prof_txt,
            snapshot=snapshot_text(req.snapshot),
        )
        messages = [
            *_history_messages(req.history),
            {"role": "user", "content": f"Write the {req.purpose} message."},
        ]
        out, _, _ = await self._grounded(f"compose:{req.purpose}", system, messages, req.snapshot, profile)
        if out is None or not out.reply:
            return render_template(req.purpose, req.snapshot)
        return out.reply
