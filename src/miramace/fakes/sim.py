"""B6 Engine — scenario simulators driven by PersonaSpec.sim (the "world" on the other side).

* SimCallerPort(personas, inner, clock): CallerPort wrapper that applies PersonaSim behaviours BEFORE
  delegating to the inner port (TextCallerPort with fake or LLM brains):
    - unknown party (no persona)                  -> NO_ANSWER
    - sim.no_answer_attempts = N                  -> the first N calls to that party are NO_ANSWER
    - sim.ghost_after = commit | order_sent       -> supplier stops answering after SUPPLIER_COMMITTED /
                                                     ORDER_SENT to it (learned via observe(event))
    - sim.refuses (pcp)                           -> request_order is REFUSED
  It also exposes `effective(party_id)`: the persona the inner brains/extractor should use for the next call,
  i.e. the YAML persona plus cooperative DEFAULT_FACTS for engine slots the YAML does not define, and (when
  sim.lost_first_request and the first request was lost) a "never received" answer to request_received.
* InboundSimulator(personas, clock): InboundPort (faxes/documents). Visit note / signed SWO after
  ORDER_REQUESTED, ADDENDUM after ADDENDUM_REQUESTED, corrected SWO after CORRECTION_REQUESTED, DELIVERY_POD
  at the slot after DELIVERY_SCHEDULED (unless the supplier ghosts).
* FactSheetDocReader(personas): DocReaderPort returning scripted note/addendum claims (quotes must be
  substrings of the doc text, else dropped) and SWO claims from the structured fax meta.
* LLMDocReader(llm): live DocReaderPort — LCDReader for notes, structured meta for SIGNED_SWO.

observe() is idempotent per event_id so a run loop may replay the whole log (resume) safely.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime
from typing import Any

from miramace.clock import add_business_hours
from miramace.events import Event, EventType
from miramace.extract.extractor import normalize
from miramace.extract.lcd import LCDReader
from miramace.models import (
    PCP_PARTY_ID,
    SWO,
    CallResult,
    CallTask,
    Claim,
    Fact,
    InboundDoc,
    InboundDocKind,
    LCDCriterion,
    PersonaSpec,
    Utterance,
    VisitNote,
)

ET = EventType

# Cooperative answers for engine slots a scenario persona does not define (slot id -> (value, text)).
DEFAULT_FACTS: dict[str, dict[str, tuple[str, str]]] = {
    "pcp": {
        "promised_action": (
            "fax visit note and signed order",
            "We'll fax the visit note and the signed order.",
        ),
        "promised_by": ("tomorrow", "By end of day tomorrow."),
        "named_owner": ("front desk", "The front desk is handling it."),
        "request_received": ("yes", "Yes, we have your request."),
        "order_status": ("in progress", "It's in progress."),
        "will_send_addendum": ("yes", "Yes, Dr. Chen will add an addendum and we'll fax it."),
        "will_send_correction": ("yes", "Yes, we'll fax a corrected signed order."),
        "will_fax_order": ("yes", "Sure, we'll fax the signed order and visit note over to them today."),
    },
    "supplier": {
        "reserve_unit": ("yes", "Yes, we can hold a K0001 chair for her."),
        "reference_number": ("", ""),  # filled per supplier below
        "order_received": ("yes", "Yes, we have the signed order from Dr. Chen's office."),
        "delivery_slot": ("tomorrow 10am", "We can bring it out tomorrow at 10 AM."),
        "delivered": ("yes", "Yes, our driver delivered it and she signed for it."),
        "k0001_in_stock": ("yes", "Yes, we have standard K0001 chairs in stock."),
        "accepts_assignment_all_months": ("yes", "Yes, we accept assignment for every rental month."),
    },
}

NEVER_GOT_IT = ("no", "We never received any request for her.")


def _ref_for(sid: str) -> str:
    initials = "".join(w[0] for w in sid.split("-") if w)[:4].upper()
    return f"{initials}-{sum(map(ord, sid)) % 9000 + 1000}"


def with_defaults(spec: PersonaSpec) -> PersonaSpec:
    """Persona plus default facts for engine slots it does not cover."""
    have = {f.slot for f in spec.facts}
    extra: list[Fact] = []
    for slot, (value, text) in DEFAULT_FACTS[spec.role].items():
        if slot in have:
            continue
        if slot == "reference_number":
            ref = _ref_for(spec.party_id)
            value, text = ref, f"Your reference number is {ref}."
        extra.append(
            Fact(id=f"default_{slot}", slot=slot, value=value, text=text, reveal_when=slot.replace("_", " "))
        )
    if not extra:
        return spec
    return spec.model_copy(update={"facts": [*spec.facts, *extra]})


def _no_answer(task: CallTask, reason: str) -> CallResult:
    return CallResult(
        task_id=task.task_id,
        outcome="NO_ANSWER",
        transcript=[
            Utterance(turn=0, speaker="system", name="system", text=f"No answer ({reason}).", ts=0.0)
        ],
        claims=[],
        flags=[f"sim:{reason}"],
        duration_s=45,
        callback_after=None,
        transport="fake",
    )


def _refused(task: CallTask, spec: PersonaSpec) -> CallResult:
    return CallResult(
        task_id=task.task_id,
        outcome="REFUSED",
        transcript=[
            Utterance(
                turn=0,
                speaker="counterparty",
                name=spec.answerer_name,
                text=f"{spec.org_name}, this is {spec.answerer_name}.",
                ts=0.0,
            ),
            Utterance(turn=1, speaker="caller", name="Mira Mace AI", text=task.purpose, ts=6.0),
            Utterance(
                turn=2,
                speaker="counterparty",
                name=spec.answerer_name,
                text="Dr. Chen won't be signing a wheelchair order for her.",
                ts=12.0,
                event="hangup",
            ),
        ],
        claims=[],
        flags=["sim:refused"],
        duration_s=18,
        callback_after=None,
        transport="fake",
    )


class _World:
    """Shared observed-world state (per case) for the simulators."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def first_time(self, e: Event) -> bool:
        if e.event_id in self.seen:
            return False
        self.seen.add(e.event_id)
        return True


class SimCallerPort:
    def __init__(self, personas: dict[str, PersonaSpec], inner: Any, clock: Any = None) -> None:
        self.personas = personas
        self.inner = inner
        self.clock = clock
        self._world = _World()
        self.calls: dict[str, int] = {}  # party_id -> calls attempted
        self.ghosted: set[str] = set()
        self.request_versions: set[int] = set()
        self.lost_versions: set[int] = set()

    # ---- world
    def observe(self, e: Event) -> None:
        if not self._world.first_time(e):
            return
        p = e.payload
        if e.type == ET.SUPPLIER_COMMITTED:
            spec = self.personas.get(p["sid"])
            if spec and spec.sim.ghost_after == "commit":
                self.ghosted.add(p["sid"])
        elif e.type == ET.ORDER_SENT:
            spec = self.personas.get(p["to_sid"])
            if spec and spec.sim.ghost_after in ("commit", "order_sent"):
                self.ghosted.add(p["to_sid"])
        elif e.type == ET.ORDER_REQUESTED:
            v = int(p["version"])
            self.request_versions.add(v)
            pcp = self.personas.get(PCP_PARTY_ID)
            if pcp and pcp.sim.lost_first_request and v == 1:
                self.lost_versions.add(v)

    def effective(self, party_id: str) -> PersonaSpec | None:
        spec = self.personas.get(party_id)
        if spec is None:
            return None
        spec = with_defaults(spec)
        latest = max(self.request_versions, default=0)
        if party_id == PCP_PARTY_ID and latest in self.lost_versions:
            facts = [f for f in spec.facts if f.slot != "request_received"]
            value, text = NEVER_GOT_IT
            facts.append(
                Fact(
                    id="sim_never_got_it",
                    slot="request_received",
                    value=value,
                    text=text,
                    reveal_when="asked whether the order request was received",
                )
            )
            spec = spec.model_copy(update={"facts": facts})
        elif spec.behaviour == "never_got_it":
            # Live-run fix: the LLM persona's "never got it" card must follow the simulated truth (only a
            # lost request version is "never received"); otherwise the office behaves helpfully.
            spec = spec.model_copy(update={"behaviour": "helpful"})
        return spec

    # ---- port
    async def run_call(self, task: CallTask) -> CallResult:
        pid = task.party_id
        spec = self.personas.get(pid)
        n = self.calls.get(pid, 0) + 1
        self.calls[pid] = n
        if spec is None:
            return _no_answer(task, "no_persona")
        if n <= spec.sim.no_answer_attempts:
            return _no_answer(task, f"no_answer_attempt_{n}")
        if pid in self.ghosted:
            return _no_answer(task, "ghosted")
        if spec.role == "pcp" and spec.sim.refuses and task.intent == "request_order":
            return _refused(task, spec)
        return await self.inner.run_call(task)


class EffectivePersonas(Mapping[str, PersonaSpec]):
    """Read-only mapping view: party_id -> SimCallerPort.effective(party_id) (for brains / FakeExtractor)."""

    def __init__(self, sim: SimCallerPort) -> None:
        self.sim = sim

    def __getitem__(self, key: str) -> PersonaSpec:
        spec = self.sim.effective(key)
        if spec is None:
            raise KeyError(key)
        return spec

    def __iter__(self) -> Iterator[str]:
        return iter(self.sim.personas)

    def __len__(self) -> int:
        return len(self.sim.personas)


# ------------------------------------------------------------------------------------------ inbound

_SWO_LABELS = {
    "beneficiary_name": "Beneficiary",
    "mbi": "Medicare number (MBI)",
    "order_date": "Order date",
    "item_description": "Item",
    "hcpcs": "HCPCS",
    "quantity": "Quantity",
    "practitioner_name": "Ordering practitioner",
    "practitioner_npi": "Practitioner NPI",
    "signed": "Signed",
    "signature_date": "Signature date",
    "accessories": "Accessories",
}

DEFAULT_SWO: dict[str, Any] = {
    "beneficiary_name": "Eleanor Martinez",
    "mbi": "1EG4-TE5-MK72",
    "order_date": "2026-11-03",
    "item_description": "Standard manual wheelchair",
    "hcpcs": "K0001",
    "quantity": 1,
    "practitioner_name": "Dr. Sarah Chen",
    "practitioner_npi": "1234567893",
    "signed": True,
    "signature_date": "2026-11-03",
    "accessories": [],
}


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "none"
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def swo_lines(swo: dict[str, Any]) -> dict[str, str]:
    """field -> the exact fax line for that field (only fields present and not None)."""
    return {k: f"{label}: {_fmt_value(swo[k])}" for k, label in _SWO_LABELS.items() if swo.get(k) is not None}


def render_swo(swo: dict[str, Any], *, corrected: bool = False) -> str:
    head = "STANDARD WRITTEN ORDER (corrected)" if corrected else "STANDARD WRITTEN ORDER"
    return "\n".join([head, "Sunrise Family Medicine", *swo_lines(swo).values()])


class InboundSimulator:
    def __init__(self, personas: dict[str, PersonaSpec], clock: Any = None) -> None:
        self.personas = personas
        self.clock = clock
        self._world = _World()
        self.queue: dict[str, list[InboundDoc]] = {}
        self.delivered_ids: set[str] = set()
        self.ghosted: set[str] = set()
        self.scheduled_kinds: dict[str, set[str]] = {}
        self.counter: dict[str, int] = {}

    @property
    def pcp(self) -> PersonaSpec | None:
        return self.personas.get(PCP_PARTY_ID)

    def _push(
        self, case_id: str, kind: InboundDocKind, at: datetime, text: str, meta: dict[str, Any]
    ) -> None:
        n = self.counter.get(f"{case_id}:{kind}", 0) + 1
        self.counter[f"{case_id}:{kind}"] = n
        doc_id = f"{case_id}:{kind.value.lower()}:{n}"
        if doc_id in self.delivered_ids:
            return
        self.queue.setdefault(case_id, []).append(
            InboundDoc(kind=kind, doc_id=doc_id, arrives_at=at, text=text, meta=meta)
        )
        self.queue[case_id].sort(key=lambda d: (d.arrives_at, d.doc_id))

    def observe(self, e: Event) -> None:
        if not self._world.first_time(e):
            return
        cid, p, t = e.case_id, e.payload, e.sim_time
        if e.source.kind == "INBOUND_DOC":  # a doc was ingested (also on replay): never deliver it again
            self.delivered_ids.add(e.source.ref)
            self.queue[cid] = [d for d in self.queue.get(cid, []) if d.doc_id != e.source.ref]
        pcp = self.pcp
        if e.type == ET.SUPPLIER_COMMITTED:
            spec = self.personas.get(p["sid"])
            if spec and spec.sim.ghost_after == "commit":
                self.ghosted.add(p["sid"])
        elif e.type == ET.ORDER_SENT:
            spec = self.personas.get(p["to_sid"])
            if spec and spec.sim.ghost_after in ("commit", "order_sent"):
                self.ghosted.add(p["to_sid"])
        elif e.type == ET.ORDER_REQUESTED and pcp is not None:
            if pcp.sim.lost_first_request and int(p["version"]) == 1:
                return
            kinds = self.scheduled_kinds.setdefault(cid, set())
            if pcp.sim.visit_note_text and "note" not in kinds:
                kinds.add("note")
                self._push(
                    cid,
                    InboundDocKind.VISIT_NOTE,
                    add_business_hours(t, pcp.sim.note_delay_bh),
                    pcp.sim.visit_note_text,
                    {"party_id": PCP_PARTY_ID},
                )
            if pcp.sim.swo_signed is not None and "swo" not in kinds:
                kinds.add("swo")
                swo = {k: v for k, v in pcp.sim.swo_signed.items() if k != "corrected"}
                self._push(
                    cid,
                    InboundDocKind.SIGNED_SWO,
                    add_business_hours(t, pcp.sim.signed_delay_bh),
                    render_swo(swo),
                    {"party_id": PCP_PARTY_ID, "swo": swo},
                )
        elif e.type == ET.ADDENDUM_REQUESTED and pcp is not None and pcp.sim.addendum_text:
            self._push(
                cid,
                InboundDocKind.ADDENDUM,
                add_business_hours(t, pcp.sim.note_delay_bh),
                pcp.sim.addendum_text,
                {"party_id": PCP_PARTY_ID, "missing": p.get("missing", [])},
            )
        elif e.type == ET.CORRECTION_REQUESTED and pcp is not None:
            base = {k: v for k, v in (pcp.sim.swo_signed or DEFAULT_SWO).items() if k != "corrected"}
            fix = (pcp.sim.swo_signed or {}).get("corrected")
            if isinstance(fix, dict):
                swo = {**base, **fix}
            else:  # fill the requested fields from a complete order
                swo = dict(base)
                for f in p.get("fields", []):
                    if f in DEFAULT_SWO:
                        swo[f] = DEFAULT_SWO[f]
            self._push(
                cid,
                InboundDocKind.SIGNED_SWO,
                add_business_hours(t, pcp.sim.fax_delay_bh),
                render_swo(swo, corrected=True),
                {"party_id": PCP_PARTY_ID, "swo": swo, "corrected": True},
            )
        elif e.type == ET.DELIVERY_SCHEDULED:
            sid = p["sid"]
            if sid in self.ghosted or sid not in self.personas:
                return
            slot = datetime.fromisoformat(p["slot"])
            org = self.personas[sid].org_name
            self._push(
                cid,
                InboundDocKind.DELIVERY_POD,
                max(slot, t),
                f"PROOF OF DELIVERY\n{org}\nK0001 standard manual wheelchair delivered and signed for.",
                {"sid": sid, "party_id": sid},
            )

    def next_arrival(self, case_id: str) -> datetime | None:
        q = self.queue.get(case_id)
        return q[0].arrives_at if q else None

    def pop_due(self, case_id: str, now: datetime) -> list[InboundDoc]:
        q = self.queue.get(case_id, [])
        due = [d for d in q if d.arrives_at <= now]
        self.queue[case_id] = [d for d in q if d.arrives_at > now]
        for d in due:
            self.delivered_ids.add(d.doc_id)
        return due


# ------------------------------------------------------------------------------------------ doc readers


def swo_claims(doc: InboundDoc) -> list[Claim]:
    swo = doc.meta.get("swo") or {}
    lines = swo_lines(swo)
    out = []
    for k, line in lines.items():
        out.append(Claim(field=k, status="stated", value=_fmt_value(swo[k]), evidence_quote=line))
    return out


def _quoted(claims: list[Claim], text: str) -> list[Claim]:
    nt = normalize(text)
    return [c for c in claims if c.evidence_quote and normalize(c.evidence_quote) in nt]


class FactSheetDocReader:
    def __init__(self, personas: Mapping[str, PersonaSpec]) -> None:
        self.personas = personas

    async def read_doc(self, doc: InboundDoc) -> list[Claim]:
        pcp = self.personas.get(PCP_PARTY_ID)
        if doc.kind == InboundDocKind.VISIT_NOTE:
            return _quoted(list(pcp.sim.note_claims) if pcp else [], doc.text)
        if doc.kind == InboundDocKind.ADDENDUM:
            return _quoted(list(pcp.sim.addendum_claims) if pcp else [], doc.text)
        if doc.kind == InboundDocKind.SIGNED_SWO:
            return swo_claims(doc)
        return []


class LLMDocReader:
    """Live doc reader: LCD checker (LLM, quotes verified in code) for notes; SWO fax is structured meta."""

    def __init__(self, llm: Any) -> None:
        self.lcd = LCDReader(llm)

    async def read_doc(self, doc: InboundDoc) -> list[Claim]:
        if doc.kind in (InboundDocKind.VISIT_NOTE, InboundDocKind.ADDENDUM):
            note = VisitNote(
                doc_id=doc.doc_id,
                text=doc.text,
                received_at=doc.arrives_at,
                is_addendum=doc.kind == InboundDocKind.ADDENDUM,
            )
            checklist = await self.lcd.check(note)
            out = []
            for crit in LCDCriterion:
                res = checklist.results.get(crit, "not_documented")
                q = checklist.evidence.get(crit)
                if res == "not_documented" or not q:
                    out.append(Claim(field=crit.value, status="unknown", value=None, evidence_quote=None))
                else:
                    out.append(Claim(field=crit.value, status="stated", value=res, evidence_quote=q))
            return out
        if doc.kind == InboundDocKind.SIGNED_SWO:
            return swo_claims(doc)
        return []


def swo_from_claims(claims: list[Claim]) -> SWO:
    """Build an SWO from (verified) claims — value parsing in code."""
    d: dict[str, Any] = {}
    for c in claims:
        if c.status != "stated" or c.value is None:
            continue
        v = c.value.strip()
        if c.field in ("order_date", "signature_date"):
            try:
                d[c.field] = date.fromisoformat(v[:10])
            except ValueError:
                continue
        elif c.field == "quantity":
            try:
                d[c.field] = int(v)
            except ValueError:
                continue
        elif c.field == "signed":
            d[c.field] = v.lower() in ("yes", "true", "1", "signed")
        elif c.field == "accessories":
            d[c.field] = [] if v.lower() in ("none", "") else [x.strip() for x in v.split(",")]
        elif c.field in SWO.model_fields:
            d[c.field] = v
    return SWO(**d)
