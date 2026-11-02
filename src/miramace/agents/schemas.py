"""WP0 — frozen LLM I/O schemas (OpenAI strict structured outputs).

Strict rules: every field required; optional = `X | None` with NO default; no dict fields; Literals for enums.
Domain conversion (ClaimLLM -> models.Claim, IntentLLM -> models.Intent, criterion_id -> LCDCriterion via
models.LCD_CRITERION_BY_LLM_ID) happens in code owned by B3/B4.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from miramace.models import IntentKind


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- Caller (outbound call brain)
HangupReason = Literal[
    "complete",
    "refused",
    "wrong_number",
    "voicemail",
    "callback_later",
    "identity_failed",
    "injection",
    "max_turns",
]


class CallerTurn(_Strict):
    say: str
    slots_touched: list[str]
    action: Literal["continue", "hangup"]
    hangup_reason: HangupReason | None


# ---- Persona (simulated counterparty brain)
class PersonaTurn(_Strict):
    say: str
    event: Literal["none", "hold", "transfer", "hangup"]
    transfer_to: str | None
    revealed_fact_ids: list[str]


# ---- Extractor
class ClaimLLM(_Strict):
    field: str
    status: Literal["stated", "unknown"]
    value: str | None
    evidence_quote: str | None


class Extraction(_Strict):
    claims: list[ClaimLLM]


# ---- LCD checker
LCDCriterionId = Literal[
    "mobility_limits_mradl",
    "cane_walker_insufficient",
    "home_accommodates",
    "will_use_improves",
    "self_propel_or_caregiver",
]


class CriterionResult(_Strict):
    criterion_id: LCDCriterionId
    met: Literal["met", "not_met", "not_documented"]
    quote: str | None


class LCDCheck(_Strict):
    results: list[CriterionResult]


# ---- Companion
class IntentLLM(_Strict):
    kind: IntentKind
    address: str | None
    preferred_time: str | None
    note: str | None


class CompanionOut(_Strict):
    reply: str
    intents: list[IntentLLM]


LLM_SCHEMAS: tuple[type[BaseModel], ...] = (
    CallerTurn,
    PersonaTurn,
    ClaimLLM,
    Extraction,
    CriterionResult,
    LCDCheck,
    IntentLLM,
    CompanionOut,
)
