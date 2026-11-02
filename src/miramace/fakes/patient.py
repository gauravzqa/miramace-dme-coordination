"""B4 — ScriptedPatient: offline PatientChannel driven by a per-purpose script.

script: {purpose: [intent dicts]} — every delivery of that purpose is answered with those intents
(e.g. {"welcome_intake": [{"kind": "confirm_address", "address": "..."}, {"kind": "give_consent"}]}).
Alternatively a list of reply dicts {"text": ..., "intents": [...]} is consumed one per delivery (last one
repeats); {"silent": true} entries mean that delivery goes unanswered. Purpose missing → None (unanswered).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from miramace.models import Intent, PatientReply

DEFAULT_UTTERANCE = {
    "confirm_address": "My address is {address}.",
    "give_consent": "Yes, please go ahead and arrange it for me.",
    "ask_status": "Where is my wheelchair?",
    "cost_concern": "I'm worried I can't afford that.",
    "cost_accepted": "Yes, that cost works for me.",
    "confirm_delivery_slot": "Yes, that time works.",
    "reschedule": "Can we move the delivery?",
    "needs_human": "Can I talk to a real person?",
    "delivery_confirmed": "Yes, it arrived and it works.",
}


class ScriptedPatient:
    def __init__(
        self, script: dict[str, list[dict[str, Any]]] | None = None, conversation_id: str = "patient"
    ):
        self.script = script or {}
        self.conversation_id = conversation_id
        self.inbox: list[tuple[str, str, datetime]] = []  # (purpose, text, at)
        self._cursor: dict[str, int] = {}

    def _intent(self, d: dict[str, Any], conv: str) -> Intent:
        kind = d["kind"]
        utt = d.get("utterance") or DEFAULT_UTTERANCE.get(kind, kind).format(address=d.get("address") or "")
        return Intent(
            kind=kind,
            address=d.get("address"),
            preferred_time=d.get("preferred_time"),
            note=d.get("note"),
            utterance=utt,
            conversation_id=conv,
        )

    async def deliver(self, case_id: str, purpose: str, text: str, at: datetime) -> PatientReply | None:
        self.inbox.append((purpose, text, at))
        entries = self.script.get(purpose)
        if not entries:
            return None
        conv = f"{self.conversation_id}:{case_id}:{purpose}:{len(self.inbox)}"
        if all("kind" in e for e in entries):  # one reply carrying all intents
            intents = [self._intent(e, conv) for e in entries]
            return PatientReply(
                conversation_id=conv, text=" ".join(i.utterance for i in intents), intents=intents, at=at
            )
        i = self._cursor.get(purpose, 0)
        self._cursor[purpose] = i + 1
        entry = entries[min(i, len(entries) - 1)]
        if entry.get("silent"):
            return None
        intents = [self._intent(e, conv) for e in entry.get("intents", [])]
        reply_text = entry.get("text") or " ".join(x.utterance for x in intents) or "OK."
        return PatientReply(conversation_id=conv, text=reply_text, intents=intents, at=at)
