"""B4 — KeywordCompanion: offline CompanionPort. Template replies built from the snapshot, keyword intents."""

from __future__ import annotations

import re

from miramace.companion.agent import HANDOFF_SENTENCE, safety_net_kind
from miramace.models import CompanionReply, ComposeRequest, Intent, Snapshot, Utterance
from miramace.notify import render_template

_ADDRESS_RE = re.compile(r"\d+\s+[^,\n]+,\s*[^,\n]+,\s*[A-Za-z]{2}\.?\s+\d{5}(?:-\d{4})?")
_COST_RE = re.compile(r"afford|expensive|cost too|too much|can't pay|cannot pay|costs too", re.I)
_STATUS_RE = re.compile(r"\b(where|status|when|update|what's happening|how long)\b", re.I)
_YES_RE = re.compile(r"\b(yes|yeah|yep|okay|ok|sure|fine|sounds good|that works|go ahead|please do)\b", re.I)
_CONSENT_RE = re.compile(r"\b(go ahead|you can|permission|consent|i agree|ok to)\b", re.I)

YES_INTENT_BY_PURPOSE = {
    "welcome_intake": "give_consent",
    "cost_disclosure": "cost_accepted",
    "delivery_slot": "confirm_delivery_slot",
    "delivered_checkin": "delivery_confirmed",
}


def status_reply(snapshot: Snapshot) -> str:
    parts = [snapshot.headline]
    for it in snapshot.items:
        if it.due_label and it.owner_label not in ("our team", "you"):
            parts.append(f"{it.owner_label[0].upper()}{it.owner_label[1:]} expects to finish {it.due_label}.")
    if snapshot.next_patient_action:
        parts.append(snapshot.next_patient_action)
    return " ".join(dict.fromkeys(parts))


class KeywordCompanion:
    def __init__(self, conversation_id: str | None = None) -> None:
        self.conversation_id = conversation_id
        self.last_purpose: str | None = None

    async def compose(self, req: ComposeRequest) -> str:
        self.last_purpose = req.purpose
        return render_template(req.purpose, req.snapshot)

    async def respond(self, snapshot: Snapshot, history: list[Utterance], text: str) -> CompanionReply:
        conv = self.conversation_id or f"conv-{snapshot.case_id}"
        intents: list[Intent] = []
        flags: list[str] = []
        replies: list[str] = []

        def add(kind: str, **kw: str | None) -> None:
            intents.append(Intent(kind=kind, utterance=text, conversation_id=conv, **kw))

        m = _ADDRESS_RE.search(text)
        if m:
            add("confirm_address", address=m.group(0))
            replies.append("Thank you, I've noted your address.")
            if _CONSENT_RE.search(text):
                add("give_consent")
        if _COST_RE.search(text):
            add("cost_concern")
            replies.append(
                "I hear you, and cost matters. Nothing moves forward on cost until you're comfortable."
            )
        kind = safety_net_kind(text)
        if kind:
            add("needs_human", note=kind)
            flags.append("safety_net_needs_human")
        if not intents and _STATUS_RE.search(text):
            add("ask_status")
            replies.append(status_reply(snapshot))
        if not intents and _YES_RE.search(text):
            k = YES_INTENT_BY_PURPOSE.get(self.last_purpose or "")
            if k:
                add(k)
                replies.append("Thank you, I've noted that.")
        if any(i.kind in ("needs_human", "cost_concern") for i in intents):
            replies.append(HANDOFF_SENTENCE)
        if not replies:
            replies.append("Thanks for letting me know. I'll find out more and let you know.")
        return CompanionReply(reply=" ".join(replies), intents=intents, grounding_ok=True, flags=flags)
