"""WP0 — frozen event contracts: EventType, one payload model per type, PAYLOADS registry, Event envelope,
make_event(), and PlannerDecision.

Event.payload is stored as a JSON-mode dict (datetimes as ISO strings). Use `event.parsed()` to get the
typed payload model back. Event.seq is assigned by store.append (0 until then).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from miramace.models import (
    PCP,
    SWO,
    Address,
    CallOutcome,
    CostInfo,
    Escalation,
    LCDChecklist,
    LCDCriterion,
    Patient,
    SupplierFacts,
    TaskSpec,
    VisitNote,
)


class EventType(StrEnum):
    # Case
    CASE_OPENED = "CASE_OPENED"
    CASE_PAUSED = "CASE_PAUSED"
    CASE_RESUMED = "CASE_RESUMED"
    CASE_CANCELLED = "CASE_CANCELLED"
    COVERAGE_CHANGED = "COVERAGE_CHANGED"
    CASE_DONE = "CASE_DONE"
    # Patient
    PATIENT_NOTIFIED = "PATIENT_NOTIFIED"
    ADDRESS_CONFIRMED = "ADDRESS_CONFIRMED"
    CONSENT_GIVEN = "CONSENT_GIVEN"
    COST_DISCLOSED = "COST_DISCLOSED"
    COST_ACCEPTED = "COST_ACCEPTED"
    CONCERN_RAISED = "CONCERN_RAISED"
    STATUS_ASKED = "STATUS_ASKED"
    PATIENT_UNREACHABLE = "PATIENT_UNREACHABLE"
    # Order
    SWO_DRAFTED = "SWO_DRAFTED"
    ORDER_REQUESTED = "ORDER_REQUESTED"
    ORDER_REQUEST_VERIFIED = "ORDER_REQUEST_VERIFIED"
    VISIT_NOTE_RECEIVED = "VISIT_NOTE_RECEIVED"
    LCD_EVALUATED = "LCD_EVALUATED"
    ADDENDUM_REQUESTED = "ADDENDUM_REQUESTED"
    SWO_SIGNED_RECEIVED = "SWO_SIGNED_RECEIVED"
    SWO_EVALUATED = "SWO_EVALUATED"
    CORRECTION_REQUESTED = "CORRECTION_REQUESTED"
    ORDER_VALIDATED = "ORDER_VALIDATED"
    ORDER_SENT = "ORDER_SENT"
    ORDER_RECEIVED = "ORDER_RECEIVED"
    ORDER_REDIRECT = "ORDER_REDIRECT"
    ORDER_STALLED = "ORDER_STALLED"
    PCP_REFUSED = "PCP_REFUSED"
    # Supplier
    CANDIDATES_RANKED = "CANDIDATES_RANKED"
    SUPPLIER_NO_ANSWER = "SUPPLIER_NO_ANSWER"
    SUPPLIER_QUALIFIED = "SUPPLIER_QUALIFIED"
    SUPPLIER_DISQUALIFIED = "SUPPLIER_DISQUALIFIED"
    SUPPLIER_COMMITTED = "SUPPLIER_COMMITTED"
    SUPPLIER_REVERIFIED = "SUPPLIER_REVERIFIED"
    SUPPLIER_RELEASED = "SUPPLIER_RELEASED"
    SUPPLIERS_EXHAUSTED = "SUPPLIERS_EXHAUSTED"
    # Assignment
    ASSIGNMENT_CONFIRMED = "ASSIGNMENT_CONFIRMED"
    ASSIGNMENT_REFUSED = "ASSIGNMENT_REFUSED"
    # Delivery
    DELIVERY_SLOT_PROPOSED = "DELIVERY_SLOT_PROPOSED"
    DELIVERY_SCHEDULED = "DELIVERY_SCHEDULED"
    DELIVERY_MISSED = "DELIVERY_MISSED"
    DELIVERED = "DELIVERED"
    POD_CONFIRMED = "POD_CONFIRMED"
    # Ledger
    COMMITMENT_MADE = "COMMITMENT_MADE"
    COMMITMENT_KEPT = "COMMITMENT_KEPT"
    COMMITMENT_BROKEN = "COMMITMENT_BROKEN"
    # Audit
    CALL_COMPLETED = "CALL_COMPLETED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    # Escalation
    ESCALATION_OPENED = "ESCALATION_OPENED"
    HUMAN_RESOLVED = "HUMAN_RESOLVED"


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Empty(Payload):
    """Payload for events that carry no data."""


# ---- Case
class CaseOpened(Payload):
    patient: Patient
    pcp: PCP


class CasePaused(Payload):
    reason: str


class CaseCancelled(Payload):
    reason: str


class CoverageChanged(Payload):
    new: str


# ---- Patient
class PatientNotified(Payload):
    purpose: str
    text: str
    grounding_ok: bool
    answered: bool


class AddressConfirmed(Payload):
    address: Address
    home_access: str | None = None


class ConsentGiven(Payload):
    scope: str


class CostDisclosed(Payload):
    cost: CostInfo


ConcernKind = Literal["cost", "confusion", "distress", "clinical", "human"]


class ConcernRaised(Payload):
    kind: ConcernKind
    utterance: str


class StatusAsked(Payload):
    utterance: str


# ---- Order
class SwoDrafted(Payload):
    swo: SWO


class OrderRequested(Payload):
    request_ref: str
    version: int


class OrderRequestVerified(Payload):
    received: bool
    named_owner: str | None = None


class VisitNoteReceived(Payload):
    note: VisitNote


class LcdEvaluated(Payload):
    checklist: LCDChecklist
    missing: list[LCDCriterion]


class AddendumRequested(Payload):
    missing: list[LCDCriterion]


class SwoSignedReceived(Payload):
    swo: SWO


class SwoEvaluated(Payload):
    complete: bool
    missing: list[str]
    hcpcs: str | None


class CorrectionRequested(Payload):
    fields: list[str]


class OrderSent(Payload):
    to_sid: str


class OrderReceived(Payload):
    by_sid: str


class OrderRedirect(Payload):
    from_sid: str
    to_sid: str


class OrderStalled(Payload):
    followups: int


class PcpRefused(Payload):
    reason: str


# ---- Supplier
class CandidatesRanked(Payload):
    sids: list[str]  # ascending distance
    distances: dict[str, float]  # sid -> miles


class SupplierNoAnswer(Payload):
    sid: str
    attempt: int


class SupplierQualified(Payload):
    sid: str
    facts: SupplierFacts
    verified_at: datetime


class SupplierDisqualified(Payload):
    sid: str
    reason: str


class SupplierCommitted(Payload):
    sid: str
    ref: str
    expires_at: datetime


class SupplierReverified(Payload):
    sid: str
    verified_at: datetime


class SupplierReleased(Payload):
    sid: str
    reason: str


# ---- Assignment
class AssignmentConfirmed(Payload):
    sid: str


class AssignmentRefused(Payload):
    sid: str


# ---- Delivery
class DeliverySlotProposed(Payload):
    sid: str
    slot: datetime


class DeliveryScheduled(Payload):
    sid: str
    slot: datetime


class Delivered(Payload):
    sid: str
    at: datetime


# ---- Ledger
class CommitmentMade(Payload):
    cid: str
    party: str
    what: str
    due_at: datetime
    evidence: str | None = None


class CommitmentRef(Payload):
    cid: str


# ---- Audit
class CallCompleted(Payload):
    task_id: str
    transcript_id: str
    outcome: CallOutcome


class ExtractionFailed(Payload):
    task_id: str
    field: str
    reason: str


# ---- Escalation
class EscalationOpened(Payload):
    escalation: Escalation


class HumanResolved(Payload):
    eid: str
    option_id: str
    note: str | None = None
    resolver: str


ET = EventType
PAYLOADS: dict[EventType, type[BaseModel]] = {
    ET.CASE_OPENED: CaseOpened,
    ET.CASE_PAUSED: CasePaused,
    ET.CASE_RESUMED: Empty,
    ET.CASE_CANCELLED: CaseCancelled,
    ET.COVERAGE_CHANGED: CoverageChanged,
    ET.CASE_DONE: Empty,
    ET.PATIENT_NOTIFIED: PatientNotified,
    ET.ADDRESS_CONFIRMED: AddressConfirmed,
    ET.CONSENT_GIVEN: ConsentGiven,
    ET.COST_DISCLOSED: CostDisclosed,
    ET.COST_ACCEPTED: Empty,
    ET.CONCERN_RAISED: ConcernRaised,
    ET.STATUS_ASKED: StatusAsked,
    ET.PATIENT_UNREACHABLE: Empty,
    ET.SWO_DRAFTED: SwoDrafted,
    ET.ORDER_REQUESTED: OrderRequested,
    ET.ORDER_REQUEST_VERIFIED: OrderRequestVerified,
    ET.VISIT_NOTE_RECEIVED: VisitNoteReceived,
    ET.LCD_EVALUATED: LcdEvaluated,
    ET.ADDENDUM_REQUESTED: AddendumRequested,
    ET.SWO_SIGNED_RECEIVED: SwoSignedReceived,
    ET.SWO_EVALUATED: SwoEvaluated,
    ET.CORRECTION_REQUESTED: CorrectionRequested,
    ET.ORDER_VALIDATED: Empty,
    ET.ORDER_SENT: OrderSent,
    ET.ORDER_RECEIVED: OrderReceived,
    ET.ORDER_REDIRECT: OrderRedirect,
    ET.ORDER_STALLED: OrderStalled,
    ET.PCP_REFUSED: PcpRefused,
    ET.CANDIDATES_RANKED: CandidatesRanked,
    ET.SUPPLIER_NO_ANSWER: SupplierNoAnswer,
    ET.SUPPLIER_QUALIFIED: SupplierQualified,
    ET.SUPPLIER_DISQUALIFIED: SupplierDisqualified,
    ET.SUPPLIER_COMMITTED: SupplierCommitted,
    ET.SUPPLIER_REVERIFIED: SupplierReverified,
    ET.SUPPLIER_RELEASED: SupplierReleased,
    ET.SUPPLIERS_EXHAUSTED: Empty,
    ET.ASSIGNMENT_CONFIRMED: AssignmentConfirmed,
    ET.ASSIGNMENT_REFUSED: AssignmentRefused,
    ET.DELIVERY_SLOT_PROPOSED: DeliverySlotProposed,
    ET.DELIVERY_SCHEDULED: DeliveryScheduled,
    ET.DELIVERY_MISSED: Empty,
    ET.DELIVERED: Delivered,
    ET.POD_CONFIRMED: Empty,
    ET.COMMITMENT_MADE: CommitmentMade,
    ET.COMMITMENT_KEPT: CommitmentRef,
    ET.COMMITMENT_BROKEN: CommitmentRef,
    ET.CALL_COMPLETED: CallCompleted,
    ET.EXTRACTION_FAILED: ExtractionFailed,
    ET.ESCALATION_OPENED: EscalationOpened,
    ET.HUMAN_RESOLVED: HumanResolved,
}

SourceKind = Literal["TIMER", "TRANSCRIPT", "PATIENT_INTENT", "HUMAN", "INBOUND_DOC", "RULE", "SYSTEM"]


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    ref: str  # transcript_id / doc_id / conversation_id / eid / rule name / task_id
    span: tuple[int, int] | None = None  # e.g. utterance turn range in a transcript


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    case_id: str
    seq: int = 0  # assigned by store.append
    type: EventType
    payload: dict[str, Any]
    source: Source
    causation_id: str | None = None
    sim_time: datetime

    @field_validator("sim_time")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("sim_time must be tz-aware (America/Chicago)")
        return v

    def parsed(self) -> BaseModel:
        """Typed payload model for this event."""
        return PAYLOADS[self.type].model_validate(self.payload)


def make_event(
    case_id: str,
    type: EventType | str,
    payload: BaseModel | dict[str, Any] | None,
    source: Source,
    sim_time: datetime,
    causation_id: str | None = None,
) -> Event:
    """Build an Event, validating the payload against PAYLOADS[type]. Raises pydantic.ValidationError."""
    et = EventType(type)
    model = PAYLOADS[et]
    if payload is None:
        payload = {}
    if isinstance(payload, BaseModel):
        if not isinstance(payload, model):
            raise TypeError(f"{et} expects {model.__name__}, got {type_name(payload)}")
        validated = payload
    else:
        validated = model.model_validate(payload)
    return Event(
        case_id=case_id,
        type=et,
        payload=validated.model_dump(mode="json"),
        source=source,
        causation_id=causation_id,
        sim_time=sim_time,
    )


def type_name(obj: object) -> str:
    return obj.__class__.__name__


class PlannerDecision(BaseModel):
    """Output of planner.plan(state, now, policy): desired tasks + rule-derived events."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskSpec] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
