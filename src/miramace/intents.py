"""B4 Patient side — Intent API: the companion's ONLY write path into the case.

`IntentAPI.submit` validates a typed Intent against current CaseState and appends events with
Source(kind="PATIENT_INTENT", ref=conversation_id). Invalid intents are rejected with a reason; this never
raises into the chat.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from miramace.events import (
    AddressConfirmed,
    ConcernRaised,
    ConsentGiven,
    DeliveryScheduled,
    EscalationOpened,
    Event,
    EventType,
    Source,
    StatusAsked,
    make_event,
)
from miramace.models import (
    Address,
    CaseState,
    CaseStatus,
    DeliveryStatus,
    Escalation,
    EscalationLane,
    EscalationOption,
    EscalationStatus,
    EscalationTrigger,
    Intent,
    PatientStatus,
    Snapshot,
    Track,
)
from miramace.snapshot import build_snapshot


class StoreLike(Protocol):
    def state(self, case_id: str) -> CaseState: ...

    def append(self, event: Event) -> Any: ...


class ClockLike(Protocol):
    def now(self) -> datetime: ...


class IntentAck(BaseModel):
    accepted: bool
    events: list[str] = Field(default_factory=list)  # EventType values appended, in order
    reason: str | None = None


# ------------------------------------------------------------------------------------ escalations

# Local fallback, aligned with escalation.CATALOGUE option ids (B2) so EFFECTS resolve the same way.
_PATIENT_OPTIONS: dict[EscalationTrigger, tuple[str, list[tuple[str, str]], str]] = {
    EscalationTrigger.COST_CONCERN: (
        'Eleanor raised a cost concern: "{utterance}". Please call her.',
        [
            ("nurse_resolved_accepted", "Spoke with Eleanor; she accepts the cost"),
            ("nurse_resolved_continue", "Spoke with Eleanor; concern addressed, she is still deciding"),
            ("patient_declined", "Eleanor declined; cancel the case"),
        ],
        "nurse_resolved_accepted",
    ),
    EscalationTrigger.NEEDS_HUMAN: (
        'Eleanor asked for a person or seemed distressed: "{utterance}". Please call her.',
        [("nurse_resolved", "Spoke with Eleanor; continue"), ("cancel", "Cancel the case")],
        "nurse_resolved",
    ),
}


def _local_escalation(state: CaseState, trigger: EscalationTrigger, context: dict[str, Any]) -> Escalation:
    question, options, suggested = _PATIENT_OPTIONS[trigger]
    n = 1 + sum(1 for e in state.escalations.values() if e.trigger == trigger)
    return Escalation(
        eid=f"esc-{trigger.value.lower()}-{n}",
        lane=EscalationLane.PATIENT,
        trigger=trigger,
        track=Track.PATIENT,
        question=question.format(utterance=context.get("utterance", "")),
        options=[EscalationOption(option_id=o, label=lbl) for o, lbl in options],
        suggested=suggested,
        context=context,
    )


def build_patient_escalation(
    state: CaseState, trigger: EscalationTrigger, intent: Intent, snapshot: Snapshot | None = None
) -> Escalation:
    """PATIENT-lane escalation. Uses B2's escalation.build_escalation when importable, else a local copy."""
    context: dict[str, Any] = {
        "utterance": intent.utterance,
        "conversation_id": intent.conversation_id,
        "patient_state": str(state.patient.state),
    }
    if snapshot is not None:
        context["snapshot_headline"] = snapshot.headline
    if intent.note:
        context["note"] = intent.note
    try:
        from miramace.escalation import build_escalation

        esc = build_escalation(trigger, state, context)
        if esc.lane == EscalationLane.PATIENT:
            return esc
    except Exception:
        pass
    return _local_escalation(state, trigger, context)


# ---------------------------------------------------------------------------------------- address

_ADDR_RE = re.compile(
    r"^\s*(?P<line1>.+?),\s*(?P<city>[A-Za-z .'-]+?),?\s+"
    r"(?P<state>[A-Za-z]{2})\.?,?\s+(?P<zip>\d{5})(?:-\d{4})?\s*\.?$"
)


def parse_us_address(text: str) -> Address | None:
    """Parse a US address; requires a 5-digit ZIP. Returns None if unparseable."""
    t = " ".join(text.split())
    try:
        return Address.parse(t)
    except ValueError:
        pass
    m = _ADDR_RE.match(t)
    if not m:
        return None
    return Address(
        line1=m.group("line1").strip(),
        city=m.group("city").strip(),
        state=m.group("state").upper(),
        zip=m.group("zip"),
    )


_CONCERN_KINDS = {"cost", "confusion", "distress", "clinical", "human"}


class IntentAPI:
    def __init__(self, store_like: StoreLike, clock: ClockLike) -> None:
        self.store = store_like
        self.clock = clock

    def snapshot(self, case_id: str) -> Snapshot:
        return build_snapshot(self.store.state(case_id), self.clock.now())

    # ------------------------------------------------------------------------------------- submit
    def submit(self, case_id: str, intent: Intent) -> IntentAck:
        try:
            state = self.store.state(case_id)
            planned = self._plan(state, intent)
        except Exception as exc:  # never raise into the chat
            return IntentAck(accepted=False, reason=f"error: {exc}")
        if isinstance(planned, str):
            return IntentAck(accepted=False, reason=planned)
        payloads, note = planned
        src = Source(kind="PATIENT_INTENT", ref=intent.conversation_id)
        appended: list[str] = []
        now = self.clock.now()
        for et, payload in payloads:
            try:
                ev = make_event(case_id, et, payload, src, now)
                self.store.append(ev)
            except Exception as exc:
                return IntentAck(
                    accepted=bool(appended), events=appended, reason=f"{et.value} rejected: {exc}"
                )
            appended.append(et.value)
        return IntentAck(accepted=True, events=appended, reason=note)

    def _plan(
        self, state: CaseState, intent: Intent
    ) -> str | tuple[list[tuple[EventType, BaseModel | None]], str | None]:
        """Return a rejection reason, or (events to append, optional note)."""
        k = intent.kind
        ET = EventType
        case_state = state.case.state
        if case_state in (CaseStatus.CANCELLED, CaseStatus.DONE) and k not in ("ask_status", "needs_human"):
            return f"case is {case_state}"

        if k == "confirm_address":
            if not intent.address:
                return "no address given"
            addr = parse_us_address(intent.address)
            if addr is None:
                return "address must be a full US address with a 5-digit ZIP code"
            current = state.patient_info.address if state.patient_info else None
            if current == addr and "address" in state.patient_flags:
                return [], "address already on file"
            return [(ET.ADDRESS_CONFIRMED, AddressConfirmed(address=addr, home_access=intent.note))], None

        if k == "give_consent":
            if "consent" in state.patient_flags:
                return [], "consent already on file"
            return [
                (ET.CONSENT_GIVEN, ConsentGiven(scope="coordinate wheelchair order on patient's behalf"))
            ], None

        if k == "ask_status":
            return [(ET.STATUS_ASKED, StatusAsked(utterance=intent.utterance))], None

        if k == "cost_accepted":
            if state.patient.state != PatientStatus.COST_DISCLOSED:
                return f"cost has not been disclosed yet (patient state {state.patient.state})"
            return [(ET.COST_ACCEPTED, None)], None

        if k in ("cost_concern", "needs_human"):
            trigger = EscalationTrigger.COST_CONCERN if k == "cost_concern" else EscalationTrigger.NEEDS_HUMAN
            open_same = any(
                e.lane == EscalationLane.PATIENT
                and e.trigger == trigger
                and e.status == EscalationStatus.OPEN
                for e in state.escalations.values()
            )
            if open_same:
                return [], "a nurse is already following up"
            if k == "cost_concern":
                kind = "cost"
            else:
                kind = intent.note if intent.note in _CONCERN_KINDS and intent.note != "cost" else "human"
            snap = build_snapshot(state, self.clock.now())
            esc = build_patient_escalation(state, trigger, intent, snap)
            return [
                (ET.CONCERN_RAISED, ConcernRaised(kind=kind, utterance=intent.utterance)),
                (ET.ESCALATION_OPENED, EscalationOpened(escalation=esc)),
            ], None

        if k == "confirm_delivery_slot":
            if state.delivery.state != DeliveryStatus.SCHEDULING or state.delivery_slot is None:
                return "no delivery time has been proposed"
            sid = state.delivery.party_id or state.committed_sid
            if not sid:
                return "no equipment company is committed"
            return [(ET.DELIVERY_SCHEDULED, DeliveryScheduled(sid=sid, slot=state.delivery_slot))], None

        if k == "reschedule":
            proposed = state.delivery.state == DeliveryStatus.SCHEDULING and state.delivery_slot is not None
            if state.delivery.state != DeliveryStatus.SCHEDULED and not proposed:
                return "no delivery time is proposed or scheduled to move"
            # declined proposal or moving a scheduled delivery: slot cleared, planner re-proposes
            return [(ET.DELIVERY_MISSED, None)], None

        if k == "delivery_confirmed":
            if state.delivery.state != DeliveryStatus.DELIVERED:
                return "the wheelchair has not been marked delivered yet"
            return [(ET.POD_CONFIRMED, None)], None

        return f"unknown intent {k}"
