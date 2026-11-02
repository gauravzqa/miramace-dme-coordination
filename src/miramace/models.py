"""WP0 — frozen domain contracts: every entity, enum and helper constructor shared by all builders.

Rules: tz-aware datetimes (America/Chicago, see CHICAGO); enums are StrEnum; domain models may have
defaults (LLM I/O schemas in agents/schemas.py may not). CaseState is produced ONLY by reducer.py.
"""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

CHICAGO = ZoneInfo("America/Chicago")

PCP_PARTY_ID = "pcp"  # party_id used for Dr. Chen's office (tasks, personas, commitments)
TARGET_HCPCS = "K0001"


def idem_key(case_id: str, party_id: str, intent: str, version: int) -> str:
    """Canonical outbox idempotency key. Format is frozen."""
    return f"{case_id}:{party_id}:{intent}:v{version}"


def slugify(name: str) -> str:
    """'Windy City Mobility & DME' -> 'windy-city-mobility-dme'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------- enums


class Owner(StrEnum):
    US = "us"
    PCP = "pcp"
    SUPPLIER = "supplier"
    PATIENT = "patient"
    ADVOCATE = "advocate"
    NONE = "none"


class Track(StrEnum):
    CASE = "case"
    PATIENT = "patient"
    ORDER = "order"
    SUPPLIER = "supplier"
    ASSIGNMENT = "assignment"
    DELIVERY = "delivery"


class CaseStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COVERAGE_CHANGED = "COVERAGE_CHANGED"
    DONE = "DONE"


class PatientStatus(StrEnum):
    INTAKE = "INTAKE"
    CONSENTED = "CONSENTED"
    COST_DISCLOSED = "COST_DISCLOSED"
    COST_ACCEPTED = "COST_ACCEPTED"
    CONCERN_RAISED = "CONCERN_RAISED"
    DELIVERY_CONFIRMED = "DELIVERY_CONFIRMED"


class OrderStatus(StrEnum):
    NOT_SENT = "NOT_SENT"
    REQUESTED = "REQUESTED"
    DOCS_CHECK = "DOCS_CHECK"
    ADDENDUM_REQUESTED = "ADDENDUM_REQUESTED"
    SIGNED = "SIGNED"
    CORRECTION_REQUESTED = "CORRECTION_REQUESTED"
    VALID = "VALID"
    SENT_TO = "SENT_TO"
    RECEIVED_BY = "RECEIVED_BY"
    BLOCKED_ON_HUMAN = "BLOCKED_ON_HUMAN"


class SupplierTrackStatus(StrEnum):
    """Derived supplier-track state. NOT_STARTED = before CANDIDATES_RANKED (WP0 addition)."""

    NOT_STARTED = "NOT_STARTED"
    CALLING = "CALLING"
    QUALIFIED = "QUALIFIED"
    COMMITTED = "COMMITTED"
    EXHAUSTED = "EXHAUSTED"


class AssignmentStatus(StrEnum):
    UNKNOWN = "UNKNOWN"
    CONFIRMED_ALL_MONTHS = "CONFIRMED_ALL_MONTHS"
    REFUSED = "REFUSED"


class DeliveryStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    SCHEDULING = "SCHEDULING"
    SCHEDULED = "SCHEDULED"
    DELIVERED = "DELIVERED"
    POD_CONFIRMED = "POD_CONFIRMED"


class SupplierStatus(StrEnum):
    UNCONTACTED = "UNCONTACTED"
    CALLING = "CALLING"
    NO_ANSWER = "NO_ANSWER"
    QUALIFIED = "QUALIFIED"
    DISQUALIFIED = "DISQUALIFIED"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"
    UNREACHABLE = "UNREACHABLE"


class LCDCriterion(StrEnum):
    MRADL_LIMITATION = "MRADL_LIMITATION"
    CANE_WALKER_INSUFFICIENT = "CANE_WALKER_INSUFFICIENT"
    HOME_ACCOMMODATES = "HOME_ACCOMMODATES"
    WILL_USE_IMPROVES = "WILL_USE_IMPROVES"
    SELF_PROPEL_OR_CAREGIVER = "SELF_PROPEL_OR_CAREGIVER"


# agents/schemas.CriterionResult.criterion_id (LLM, lower snake) -> LCDCriterion
LCD_CRITERION_BY_LLM_ID: dict[str, LCDCriterion] = {
    "mobility_limits_mradl": LCDCriterion.MRADL_LIMITATION,
    "cane_walker_insufficient": LCDCriterion.CANE_WALKER_INSUFFICIENT,
    "home_accommodates": LCDCriterion.HOME_ACCOMMODATES,
    "will_use_improves": LCDCriterion.WILL_USE_IMPROVES,
    "self_propel_or_caregiver": LCDCriterion.SELF_PROPEL_OR_CAREGIVER,
}

LCDResult = Literal["met", "not_met", "not_documented"]


class CommitmentStatus(StrEnum):
    OPEN = "OPEN"
    KEPT = "KEPT"
    BROKEN = "BROKEN"
    CANCELLED = "CANCELLED"


class EscalationLane(StrEnum):
    CASE = "CASE"
    PATIENT = "PATIENT"


class EscalationStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"


class EscalationTrigger(StrEnum):
    LCD_MISSING_AFTER_ADDENDUM = "LCD_MISSING_AFTER_ADDENDUM"
    ORDER_STALLED = "ORDER_STALLED"
    CODE_DRIFT = "CODE_DRIFT"
    SWO_INCOMPLETE_AFTER_CORRECTION = "SWO_INCOMPLETE_AFTER_CORRECTION"
    SUPPLIERS_EXHAUSTED = "SUPPLIERS_EXHAUSTED"
    EXTRACTION_CHECK_FAILED = "EXTRACTION_CHECK_FAILED"
    COST_CONCERN = "COST_CONCERN"
    PATIENT_UNREACHABLE = "PATIENT_UNREACHABLE"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    PCP_REFUSED = "PCP_REFUSED"


class TaskKind(StrEnum):
    CALL_PCP = "CALL_PCP"
    CALL_SUPPLIER = "CALL_SUPPLIER"
    NOTIFY_PATIENT = "NOTIFY_PATIENT"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class InboundDocKind(StrEnum):
    VISIT_NOTE = "VISIT_NOTE"
    ADDENDUM = "ADDENDUM"
    SIGNED_SWO = "SIGNED_SWO"
    ORDER_FAXED = "ORDER_FAXED"
    DELIVERY_POD = "DELIVERY_POD"


NotifyPurpose = Literal[
    "welcome_intake", "cost_disclosure", "delivery_slot", "delivered_checkin", "status_update"
]

# ------------------------------------------------------------------------------------- core entities


class TrackState(_Model):
    state: str
    owner: Owner = Owner.US
    due_at: datetime | None = None
    party_id: str | None = None


class Address(_Model):
    line1: str
    city: str
    state: str
    zip: str

    def one_line(self) -> str:
        return f"{self.line1}, {self.city}, {self.state} {self.zip}"

    @classmethod
    def parse(cls, text: str) -> Address:
        """Parse '1820 N Clark St, Chicago, IL 60614'."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 3:
            raise ValueError(f"cannot parse address: {text!r}")
        m = re.fullmatch(r"([A-Z]{2})\s+(\d{5})(?:-\d{4})?", parts[-1])
        if not m:
            raise ValueError(f"cannot parse state/zip: {text!r}")
        return cls(line1=", ".join(parts[:-2]), city=parts[-2], state=m.group(1), zip=m.group(2))


class PatientProfile(_Model):
    best_hours: tuple[int, int] = (9, 20)
    caregiver_name: str | None = None
    caregiver_phone: str | None = None
    hearing_or_language: str | None = None
    prior_concerns: list[str] = Field(default_factory=list)
    home_access: str | None = None


class PatientPHI(_Model):
    name: str
    dob: date
    mbi: str


class Patient(_Model):
    name: str
    age: int
    dob: date
    mbi: str = "1EG4-TE5-MK72"
    coverage: str = "ORIGINAL_PART_B"
    medigap: bool = False
    address: Address | None = None
    profile: PatientProfile = Field(default_factory=PatientProfile)

    def phi(self) -> PatientPHI:
        return PatientPHI(name=self.name, dob=self.dob, mbi=self.mbi)


class PCP(_Model):
    name: str
    practice: str
    phone: str
    npi: str = "1234567893"
    party_id: str = PCP_PARTY_ID


class Supplier(_Model):
    sid: str
    name: str
    phone: str
    address: Address
    distance_mi: float | None = None

    @property
    def zip(self) -> str:
        return self.address.zip


class SupplierFacts(_Model):
    taking_new_medicare: bool | None = None
    k0001_in_stock: bool | None = None
    delivers_to_zip: bool | None = None
    accepts_assignment_all_months: bool | None = None
    earliest_delivery_days: int | None = None
    contact_name: str | None = None
    evidence: dict[str, str] = Field(default_factory=dict)  # field -> verbatim quote


class SupplierRecord(_Model):
    sid: str
    status: SupplierStatus = SupplierStatus.UNCONTACTED
    attempts: int = 0
    facts: SupplierFacts | None = None
    verified_at: datetime | None = None
    ref: str | None = None
    expires_at: datetime | None = None
    reason: str | None = None


class SWO(_Model):
    """Standard Written Order."""

    beneficiary_name: str | None = None
    mbi: str | None = None
    order_date: date | None = None
    item_description: str | None = None
    hcpcs: str | None = None
    quantity: int | None = None
    practitioner_name: str | None = None
    practitioner_npi: str | None = None
    signed: bool = False
    signature_date: date | None = None
    accessories: list[str] = Field(default_factory=list)


class LCDChecklist(_Model):
    results: dict[LCDCriterion, LCDResult] = Field(default_factory=dict)
    evidence: dict[LCDCriterion, str] = Field(default_factory=dict)

    @property
    def missing(self) -> list[LCDCriterion]:
        """Criteria not 'met' (absent criteria count as missing)."""
        return [c for c in LCDCriterion if self.results.get(c) != "met"]

    @property
    def all_met(self) -> bool:
        return not self.missing


class VisitNote(_Model):
    doc_id: str
    text: str
    received_at: datetime
    is_addendum: bool = False


class InboundDoc(_Model):
    kind: InboundDocKind
    doc_id: str
    arrives_at: datetime
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class Commitment(_Model):
    cid: str
    party: str  # party_id ("pcp" or a supplier sid)
    what: str
    due_at: datetime
    status: CommitmentStatus = CommitmentStatus.OPEN
    source_event_id: str | None = None
    followups: int = 0


class EscalationOption(_Model):
    option_id: str
    label: str


class Escalation(_Model):
    eid: str
    lane: EscalationLane
    trigger: EscalationTrigger
    track: Track
    question: str
    options: list[EscalationOption]
    suggested: str | None = None  # option_id
    context: dict[str, Any] = Field(default_factory=dict)
    status: EscalationStatus = EscalationStatus.OPEN
    resolution: str | None = None  # option_id chosen


class TaskSpec(_Model):
    """A desired task as emitted by the planner (Task minus task_id/case_id/status/attempts)."""

    kind: TaskKind
    party_id: str
    intent: str
    version: int = 1
    idem_key: str
    due_at: datetime
    max_attempts: int = 3
    args: dict[str, Any] = Field(default_factory=dict)


class Task(_Model):
    task_id: str
    case_id: str
    kind: TaskKind
    party_id: str
    intent: str
    version: int = 1
    idem_key: str
    due_at: datetime
    attempts: int = 0
    max_attempts: int = 3
    status: TaskStatus = TaskStatus.PENDING
    args: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------- calls (§2 frozen)


class SlotSpec(_Model):
    id: str
    ask: str
    type: Literal["bool", "int", "str", "date"]
    required: bool


class CallTask(_Model):
    task_id: str
    party: Literal["pcp", "supplier"]
    party_id: str
    org_name: str
    phone: str
    intent: str
    purpose: str
    slots: list[SlotSpec]
    phi: PatientPHI | None = None  # None until code allows it (PHI gate)
    context: dict[str, str] = Field(default_factory=dict)
    max_turns: int = 14


class Utterance(_Model):
    turn: int
    speaker: Literal["caller", "counterparty", "system"]
    name: str
    text: str
    ts: float
    heard_text: str | None = None
    event: str | None = None  # hold / transfer / hangup


class Claim(_Model):
    field: str
    status: Literal["stated", "unknown"]
    value: str | None
    evidence_quote: str | None


class LLMCall(_Model):
    agent: str
    key: str
    model: str
    in_tok: int
    out_tok: int
    ms: int
    ok: bool


CallOutcome = Literal[
    "COMPLETED", "NO_ANSWER", "VOICEMAIL", "WRONG_NUMBER", "CALLBACK_REQUESTED", "REFUSED", "FAILED"
]
CallTransport = Literal["text", "voice", "text_fallback", "fake"]


class CallResult(_Model):
    task_id: str
    outcome: CallOutcome
    transcript: list[Utterance]
    claims: list[Claim]
    flags: list[str]
    duration_s: int
    callback_after: datetime | None
    transport: CallTransport
    usage: list[LLMCall] = Field(default_factory=list)


# ------------------------------------------------------------------------------- patient (§2 frozen)

IntentKind = Literal[
    "confirm_address",
    "give_consent",
    "ask_status",
    "cost_concern",
    "cost_accepted",
    "confirm_delivery_slot",
    "reschedule",
    "needs_human",
    "delivery_confirmed",
]


class Intent(_Model):
    kind: IntentKind
    address: str | None = None
    preferred_time: str | None = None
    note: str | None = None
    utterance: str
    conversation_id: str


class SnapshotItem(_Model):
    track: Track
    plain_status: str
    owner_label: str
    due_label: str | None = None


class CostInfo(_Model):
    deductible_usd: int = 283
    coinsurance_pct: int = 20
    rental_months: int = 13


class Snapshot(_Model):
    case_id: str
    as_of: datetime
    headline: str
    items: list[SnapshotItem]
    cost: CostInfo = Field(default_factory=CostInfo)
    next_patient_action: str | None = None
    allowed_names: list[str] = Field(default_factory=list)
    allowed_dates: list[str] = Field(default_factory=list)
    allowed_amounts: list[str] = Field(default_factory=list)


class ComposeRequest(_Model):
    """Input to CompanionPort.compose: an outbound update chosen by notify policy (code)."""

    case_id: str
    purpose: NotifyPurpose
    snapshot: Snapshot
    profile: PatientProfile = Field(default_factory=PatientProfile)
    history: list[Utterance] = Field(default_factory=list)


class CompanionReply(_Model):
    """Output of CompanionPort.respond after grounding + keyword safety net."""

    reply: str
    intents: list[Intent] = Field(default_factory=list)
    grounding_ok: bool = True
    flags: list[str] = Field(default_factory=list)  # e.g. grounding_hold, safety_net_needs_human


class PatientReply(_Model):
    """What PatientChannel.deliver returns when the patient answers (None = no answer)."""

    conversation_id: str
    text: str
    intents: list[Intent] = Field(default_factory=list)  # pre-parsed (scripted patient); else empty
    at: datetime | None = None


# ----------------------------------------------------------------------------------- personas (sims)


class Fact(_Model):
    id: str
    slot: str
    value: str
    text: str
    reveal_when: str


class PersonaSim(_Model):
    no_answer_attempts: int = 0
    ghost_after: Literal["never", "commit", "order_sent"] = "never"
    lost_first_request: bool = False
    note_delay_bh: float = 16
    signed_delay_bh: float = 24
    fax_delay_bh: float = 4
    visit_note_text: str | None = None
    addendum_text: str | None = None
    swo_signed: dict[str, Any] | None = None
    note_claims: list[Claim] = Field(default_factory=list)
    addendum_claims: list[Claim] = Field(default_factory=list)
    refuses: bool = False
    callback_after_hour: int | None = None
    contradicts_on_readback: bool = False


PersonaBehaviour = Literal[
    "helpful", "partial", "ivr", "transfer", "callback_later", "never_got_it", "injection"
]


class PersonaSpec(_Model):
    party_id: str
    role: Literal["pcp", "supplier"]
    org_name: str
    answerer_name: str
    behaviour: PersonaBehaviour = "helpful"
    facts: list[Fact] = Field(default_factory=list)
    sim: PersonaSim = Field(default_factory=PersonaSim)


# ---------------------------------------------------------------------------- CaseState (projection)


class CaseState(_Model):
    """Pure projection of the event log. Built and mutated ONLY by reducer.py."""

    case_id: str
    case: TrackState = Field(default_factory=lambda: TrackState(state=CaseStatus.ACTIVE, owner=Owner.US))
    patient: TrackState = Field(
        default_factory=lambda: TrackState(state=PatientStatus.INTAKE, owner=Owner.PATIENT)
    )
    order: TrackState = Field(default_factory=lambda: TrackState(state=OrderStatus.NOT_SENT, owner=Owner.US))
    supplier: TrackState = Field(
        default_factory=lambda: TrackState(state=SupplierTrackStatus.NOT_STARTED, owner=Owner.US)
    )
    assignment: TrackState = Field(
        default_factory=lambda: TrackState(state=AssignmentStatus.UNKNOWN, owner=Owner.US)
    )
    delivery: TrackState = Field(
        default_factory=lambda: TrackState(state=DeliveryStatus.NOT_STARTED, owner=Owner.US)
    )
    patient_info: Patient | None = None
    pcp: PCP | None = None
    suppliers: dict[str, SupplierRecord] = Field(default_factory=dict)
    ranked: list[str] = Field(default_factory=list)  # sids in CANDIDATES_RANKED order
    committed_sid: str | None = None
    backup_sid: str | None = None
    assignment_by_sid: dict[str, AssignmentStatus] = Field(default_factory=dict)
    swo: SWO | None = None
    swo_missing: list[str] = Field(default_factory=list)
    lcd: LCDChecklist | None = None
    visit_notes: list[VisitNote] = Field(default_factory=list)
    addendum_requests: int = 0
    correction_requests: int = 0
    order_request_version: int = 0
    order_sent_to: str | None = None
    order_received_by: str | None = None
    delivery_slot: datetime | None = None
    commitments: dict[str, Commitment] = Field(default_factory=dict)
    escalations: dict[str, Escalation] = Field(default_factory=dict)
    last_notified_at: datetime | None = None
    notify_attempts: int = 0
    # F1 (additive): PATIENT_NOTIFIED sends per purpose (delivery_slot is reset per proposed slot) -> fresh
    # notify versions; DELIVERY_SLOT_PROPOSED count -> propose_slot versions; visit notes seen when the last
    # addendum was requested (LCD escalation only after a note arrives AFTER the request).
    notify_counts: dict[str, int] = Field(default_factory=dict)
    slot_proposals: int = 0
    addendum_notes_seen: int = 0
    pending_notify: list[str] = Field(default_factory=list)
    patient_flags: set[str] = Field(default_factory=set)  # seen: "address", "consent"
    resume_to: str | None = None  # patient state to restore after CONCERN_RAISED resolves
    last_seq: int = 0

    def track(self, t: Track | str) -> TrackState:
        return getattr(self, Track(t).value)


# ---------------------------------------------------------------------------------- constructors


def eleanor() -> Patient:
    return Patient(name="Eleanor Martinez", age=72, dob=date(1954, 3, 14), address=None)


def dr_chen() -> PCP:
    return PCP(name="Dr. Sarah Chen", practice="Sunrise Family Medicine", phone="(312) 555-0198")


_DEFAULT_SUPPLIERS = Path(__file__).resolve().parents[2] / "data" / "suppliers.csv"


def load_suppliers(path: str | Path = "data/suppliers.csv") -> list[Supplier]:
    """Load the directory CSV. Relative path falls back to the repo's data/ when cwd differs."""
    p = Path(path)
    if not p.exists() and not p.is_absolute() and str(path) == "data/suppliers.csv":
        p = _DEFAULT_SUPPLIERS
    out: list[Supplier] = []
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["supplier_name"].strip()
            out.append(
                Supplier(
                    sid=slugify(name),
                    name=name,
                    phone=row["phone"].strip(),
                    address=Address.parse(row["address"]),
                )
            )
    return out
