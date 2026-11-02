"""WP0 — frozen port Protocols (execution-plan §2). Every port is async except InboundPort (sync sim)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

from miramace.agents.schemas import CallerTurn, PersonaTurn
from miramace.events import Event
from miramace.models import (
    CallResult,
    CallTask,
    Claim,
    CompanionReply,
    ComposeRequest,
    InboundDoc,
    PatientReply,
    PersonaSpec,
    Snapshot,
    Utterance,
)

T = TypeVar("T", bound=BaseModel)

CallPhase = Literal["opening", "verify_party", "verified", "readback", "closing"]


class TranscriptView(BaseModel):
    """What a Brain sees for one turn. `me` says which side the brain plays."""

    model_config = ConfigDict(extra="forbid")

    task: CallTask
    phase: str  # one of CallPhase; computed by code (calls/runner.phase_from)
    utterances: list[Utterance]
    me: Literal["caller", "counterparty"]
    persona: PersonaSpec | None = None  # set only for the counterparty brain (hidden from caller)


@runtime_checkable
class LLMPort(Protocol):
    async def structured(
        self, *, agent: str, key: str, system: str, messages: list[dict], schema: type[T]
    ) -> T: ...


@runtime_checkable
class CallerPort(Protocol):
    async def run_call(self, task: CallTask) -> CallResult: ...


@runtime_checkable
class DocReaderPort(Protocol):
    async def read_doc(self, doc: InboundDoc) -> list[Claim]: ...


@runtime_checkable
class CompanionPort(Protocol):
    async def compose(self, req: ComposeRequest) -> str: ...

    async def respond(self, snapshot: Snapshot, history: list[Utterance], text: str) -> CompanionReply: ...


@runtime_checkable
class PatientChannel(Protocol):
    async def deliver(self, case_id: str, purpose: str, text: str, at: datetime) -> PatientReply | None: ...


@runtime_checkable
class InboundPort(Protocol):
    """Fax/document simulator driven by PersonaSpec.sim. Synchronous."""

    def observe(self, e: Event) -> None: ...

    def next_arrival(self, case_id: str) -> datetime | None: ...

    def pop_due(self, case_id: str, now: datetime) -> list[InboundDoc]: ...


@runtime_checkable
class Brain(Protocol):
    """Shared by text and voice transports; stateless function of the view."""

    async def next(self, view: TranscriptView) -> CallerTurn | PersonaTurn: ...


@dataclass
class Ports:
    caller_pcp: CallerPort
    caller_supplier: CallerPort
    doc_reader: DocReaderPort
    companion: CompanionPort
    patient: PatientChannel
    inbound: InboundPort
    llm: LLMPort | None = None  # None in pure-fake runs
