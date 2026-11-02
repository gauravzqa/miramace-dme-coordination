"""B2 — escalation catalogue (closed set of triggers), opening, and option → effect table.

* CATALOGUE[trigger] gives lane, blocked track, question template, typed options and the suggested option.
* open_escalation(...) builds an ESCALATION_OPENED event (default Source RULE). eid is deterministic:
  f"esc-{trigger.lower()}-{n}" with n = 1 + number of escalations of that trigger already in state.
* EFFECTS[(trigger, option_id)](state, now, note) -> follow-up events (Source HUMAN). Options not in
  EFFECTS have no follow-up events (manual / reducer restores the track, or the planner re-plans).
* resolve(...) = HUMAN_RESOLVED + effects, after validating the eid is open and the option belongs to it.
  The batch is dry-run through the reducer; if any effect would be illegal, only HUMAN_RESOLVED is returned
  (its note explains that the effect must be done manually) — resolve never yields an unappendable batch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from miramace.clock import add_business_hours
from miramace.events import Event, EventType, Source, make_event
from miramace.models import (
    PCP_PARTY_ID,
    SWO,
    TARGET_HCPCS,
    CaseState,
    CostInfo,
    Escalation,
    EscalationLane,
    EscalationOption,
    EscalationStatus,
    EscalationTrigger,
    Track,
)

T = EscalationTrigger
Effect = Callable[[CaseState, datetime, str | None], list[Event]]


@dataclass(frozen=True)
class TriggerSpec:
    lane: EscalationLane
    track: Track
    question: str  # str.format_map template over the escalation context (missing keys → "unknown")
    options: tuple[tuple[str, str], ...]  # (option_id, label)
    suggested: str | None


_CANCEL = ("cancel", "Cancel the case")

CATALOGUE: dict[EscalationTrigger, TriggerSpec] = {
    T.LCD_MISSING_AFTER_ADDENDUM: TriggerSpec(
        EscalationLane.CASE,
        Track.ORDER,
        "Dr. Chen's notes still do not document {missing} after an addendum request. How should we proceed?",
        (("pcp_sent_addendum", "I spoke to Dr. Chen; a further addendum is on its way"), _CANCEL),
        "pcp_sent_addendum",
    ),
    T.ORDER_STALLED: TriggerSpec(
        EscalationLane.CASE,
        Track.ORDER,
        "Dr. Chen's office has not delivered '{what}' after {followups} follow-ups. How should we proceed?",
        (
            ("advocate_called_got_owner", "I called and got a named owner with a new date"),
            ("switch_to_portal", "Submit through the practice portal manually"),
        ),
        "advocate_called_got_owner",
    ),
    T.CODE_DRIFT: TriggerSpec(
        EscalationLane.CASE,
        Track.ORDER,
        "The signed order lists HCPCS {hcpcs} instead of K0001 (standard manual wheelchair). "
        "How should we proceed?",
        (
            ("pcp_corrected_to_K0001", "Dr. Chen corrected the order to K0001"),
            ("proceed_new_code_manual", "Proceed with the new code manually (outside automation)"),
            _CANCEL,
        ),
        "pcp_corrected_to_K0001",
    ),
    T.SWO_INCOMPLETE_AFTER_CORRECTION: TriggerSpec(
        EscalationLane.CASE,
        Track.ORDER,
        "The corrected order is still missing: {missing}. How should we proceed?",
        (("pcp_sent_corrected", "I spoke to Dr. Chen; a complete order is on its way"), _CANCEL),
        "pcp_sent_corrected",
    ),
    T.SUPPLIERS_EXHAUSTED: TriggerSpec(
        EscalationLane.CASE,
        Track.SUPPLIER,
        "None of the {count} nearby suppliers could take the order ({summary}). How should we proceed?",
        (
            ("expand_radius", "Expand the search radius (manual)"),
            ("advocate_found_supplier", "I found a supplier and will arrange it manually"),
        ),
        "expand_radius",
    ),
    T.EXTRACTION_CHECK_FAILED: TriggerSpec(
        EscalationLane.CASE,
        Track.SUPPLIER,
        "We could not verify '{field}' from the call with {party}: {reason}. How should we proceed?",
        (("accept_value", "Accept the value as heard"), ("recall", "Call them back to confirm")),
        "recall",
    ),
    T.COST_CONCERN: TriggerSpec(
        EscalationLane.PATIENT,
        Track.PATIENT,
        'Eleanor raised a cost concern: "{utterance}". Please call her.',
        (
            ("nurse_resolved_accepted", "Spoke with Eleanor; she accepts the cost"),
            ("nurse_resolved_continue", "Spoke with Eleanor; concern addressed, she is still deciding"),
            ("patient_declined", "Eleanor declined; cancel the case"),
        ),
        "nurse_resolved_accepted",
    ),
    T.PATIENT_UNREACHABLE: TriggerSpec(
        EscalationLane.PATIENT,
        Track.PATIENT,
        "Eleanor has not responded to {attempts} messages about {purpose}. Please try to reach her.",
        (("nurse_reached_patient", "Reached Eleanor (or her caregiver)"), _CANCEL),
        "nurse_reached_patient",
    ),
    T.NEEDS_HUMAN: TriggerSpec(
        EscalationLane.PATIENT,
        Track.PATIENT,
        'Eleanor asked for a person or seemed distressed: "{utterance}". Please call her.',
        (("nurse_resolved", "Spoke with Eleanor; continue"), _CANCEL),
        "nurse_resolved",
    ),
    T.PCP_REFUSED: TriggerSpec(
        EscalationLane.CASE,
        Track.ORDER,
        "Dr. Chen's office declined to proceed: {reason}. How should we proceed?",
        (("advocate_resolved_with_pcp", "I spoke to Dr. Chen; they will send the order"), _CANCEL),
        "advocate_resolved_with_pcp",
    ),
}


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "unknown"


def _fmt(v: Any) -> Any:
    if isinstance(v, list | tuple | set):
        return ", ".join(str(x) for x in v) or "none"
    return v


def option_ids(trigger: EscalationTrigger | str) -> list[str]:
    return [oid for oid, _ in CATALOGUE[EscalationTrigger(trigger)].options]


def next_eid(trigger: EscalationTrigger | str, state: CaseState) -> str:
    trig = EscalationTrigger(trigger)
    n = 1 + sum(1 for e in state.escalations.values() if e.trigger == trig)
    return f"esc-{trig.value.lower()}-{n}"


def build_escalation(
    trigger: EscalationTrigger | str, state: CaseState, context: dict[str, Any] | None = None
) -> Escalation:
    trig = EscalationTrigger(trigger)
    spec = CATALOGUE[trig]
    ctx = dict(context or {})
    track = Track(ctx.get("track", spec.track))
    question = spec.question.format_map(_SafeDict({k: _fmt(v) for k, v in ctx.items()}))
    return Escalation(
        eid=next_eid(trig, state),
        lane=spec.lane,
        trigger=trig,
        track=track,
        question=question,
        options=[EscalationOption(option_id=o, label=lbl) for o, lbl in spec.options],
        suggested=spec.suggested,
        context=ctx,
    )


def open_escalation(
    trigger: EscalationTrigger | str,
    state: CaseState,
    now: datetime,
    context: dict[str, Any] | None = None,
    source: Source | None = None,
    causation_id: str | None = None,
) -> Event:
    """ESCALATION_OPENED event with a deterministic eid and event_id (default Source RULE)."""
    esc = build_escalation(trigger, state, context)
    src = source or Source(kind="RULE", ref=f"escalation:{esc.trigger.value}")
    ev = make_event(
        state.case_id,
        EventType.ESCALATION_OPENED,
        {"escalation": esc.model_dump(mode="json")},
        src,
        now,
        causation_id=causation_id,
    )
    return ev.model_copy(update={"event_id": f"{state.case_id}:{esc.eid}:opened"})


# ------------------------------------------------------------------------------------------ effects

_HUMAN = Source(kind="HUMAN", ref="escalation")


def _cancel(state: CaseState, now: datetime, note: str | None) -> list[Event]:
    reason = note or "cancelled by advocate"
    return [make_event(state.case_id, EventType.CASE_CANCELLED, {"reason": reason}, _HUMAN, now)]


def _cost_accepted(state: CaseState, now: datetime, note: str | None) -> list[Event]:
    """State-aware: if cost was never disclosed (concern raised early), the nurse's call is the disclosure."""
    out = []
    if "cost_disclosed" not in state.patient_flags:
        out.append(make_event(state.case_id, EventType.COST_DISCLOSED, {"cost": CostInfo()}, _HUMAN, now))
    out.append(make_event(state.case_id, EventType.COST_ACCEPTED, None, _HUMAN, now))
    return out


def _manual_new_code(state: CaseState, now: datetime, note: str | None) -> list[Event]:
    reason = "handled manually by advocate: non-K0001 code" + (f" ({note})" if note else "")
    return [make_event(state.case_id, EventType.CASE_CANCELLED, {"reason": reason}, _HUMAN, now)]


def _new_pcp_commitment(state: CaseState, now: datetime, note: str | None) -> list[Event]:
    payload = {
        "cid": f"human-{state.last_seq}-{PCP_PARTY_ID}",
        "party": PCP_PARTY_ID,
        "what": "send the signed order paperwork (advocate-confirmed owner)",
        "due_at": add_business_hours(now, 24),
        "evidence": note,
    }
    return [make_event(state.case_id, EventType.COMMITMENT_MADE, payload, _HUMAN, now)]


def _corrected_swo(state: CaseState, now: datetime, note: str | None) -> list[Event]:
    swo = (state.swo or SWO()).model_copy(update={"hcpcs": TARGET_HCPCS})
    return [make_event(state.case_id, EventType.SWO_SIGNED_RECEIVED, {"swo": swo}, _HUMAN, now)]


EFFECTS: dict[tuple[EscalationTrigger, str], Effect] = {
    (T.COST_CONCERN, "nurse_resolved_accepted"): _cost_accepted,
    (T.COST_CONCERN, "patient_declined"): _cancel,
    (T.CODE_DRIFT, "pcp_corrected_to_K0001"): _corrected_swo,
    (T.CODE_DRIFT, "cancel"): _cancel,
    (T.CODE_DRIFT, "proceed_new_code_manual"): _manual_new_code,
    (T.ORDER_STALLED, "advocate_called_got_owner"): _new_pcp_commitment,
    (T.PCP_REFUSED, "advocate_resolved_with_pcp"): _new_pcp_commitment,
    (T.PCP_REFUSED, "cancel"): _cancel,
    (T.LCD_MISSING_AFTER_ADDENDUM, "cancel"): _cancel,
    (T.SWO_INCOMPLETE_AFTER_CORRECTION, "cancel"): _cancel,
    (T.PATIENT_UNREACHABLE, "cancel"): _cancel,
    (T.NEEDS_HUMAN, "cancel"): _cancel,
    # No follow-up events (manual, doc arrives later, or the planner re-plans):
    # SUPPLIERS_EXHAUSTED/*, EXTRACTION_CHECK_FAILED/*, ORDER_STALLED/switch_to_portal,
    # *_pcp_sent_*, COST_CONCERN/nurse_resolved_continue, nurse_* .
}


def resolve(
    state: CaseState,
    eid: str,
    option_id: str,
    note: str | None,
    resolver: str,
    now: datetime,
) -> list[Event]:
    """HUMAN_RESOLVED + effect events. Raises ValueError for unknown/closed eid or a foreign option_id."""
    esc = state.escalations.get(eid)
    if esc is None:
        raise ValueError(f"unknown escalation {eid!r}")
    if esc.status != EscalationStatus.OPEN:
        raise ValueError(f"escalation {eid!r} is already {esc.status}")
    valid = [o.option_id for o in esc.options] or option_ids(esc.trigger)
    if option_id not in valid:
        raise ValueError(f"option {option_id!r} not valid for {esc.trigger}; choose from {valid}")
    src = Source(kind="HUMAN", ref=eid)
    hr = make_event(
        state.case_id,
        EventType.HUMAN_RESOLVED,
        {"eid": eid, "option_id": option_id, "note": note, "resolver": resolver},
        src,
        now,
    )
    hr = hr.model_copy(update={"event_id": f"{state.case_id}:{eid}:resolved"})
    out = [hr]
    effect = EFFECTS.get((esc.trigger, option_id))
    if effect is not None:
        for i, ev in enumerate(effect(state, now, note)):
            out.append(
                ev.model_copy(
                    update={
                        "event_id": f"{state.case_id}:{eid}:effect:{i}",
                        "source": src,
                        "causation_id": hr.event_id,
                    }
                )
            )
    err = _dry_run(state, out)
    if err is not None:
        msg = f"automatic follow-up not applied ({err}); handle manually"
        payload = {**hr.payload, "note": f"{note}; {msg}" if note else msg}
        return [hr.model_copy(update={"payload": payload})]
    return out


def _dry_run(state: CaseState, events: list[Event]) -> str | None:
    """Reduce `events` on a copy of `state`; return the first guard failure, else None."""
    from miramace.reducer import IllegalTransition, reduce  # local: reducer must not depend on escalation

    st = state
    for ev in events:
        try:
            st = reduce(st, ev.model_copy(update={"seq": st.last_seq + 1}))
        except IllegalTransition as exc:
            return str(exc)
    return None
