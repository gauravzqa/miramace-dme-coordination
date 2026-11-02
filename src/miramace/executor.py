"""B6 Engine — impure executor: build CallTask (PHI gate), run ports, store transcripts, interpret, append.

Executor(ports, store, clock, on_event=None):
  execute(task, state)  -> appended events (CALL_* or NOTIFY_PATIENT)
  ingest_doc(doc, state)-> appended events (fax/document arrival)
  append(events)        -> append to the store one by one (an IllegalTransition becomes EXTRACTION_FAILED so a
                           stale interpretation never crashes the loop), then notify observers + on_event.

PHI GATE (code, not the LLM): CallTask.phi is None for supplier `screen`; for other supplier intents it is
set only when that supplier is QUALIFIED/COMMITTED; PCP calls carry PHI (treatment disclosure). The runner
still withholds PHI until the organisation is verified on the call.

Retries: `screen` and `followup_order_*` tasks are always closed (the planner owns their retry ladder). Other
calls that are unanswered or teach us nothing are rescheduled (+2 business hours or the callback time) until
max_attempts, then FAILED; an exhausted supplier commit/propose_slot/verify_delivered releases that supplier
(+ORDER_REDIRECT), an exhausted PCP call opens ORDER_STALLED.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from miramace.clock import add_business_hours, is_patient_time, next_patient_time
from miramace.escalation import open_escalation
from miramace.events import Event, EventType, Source, make_event
from miramace.grounding import grounding_check
from miramace.intents import IntentAPI
from miramace.interpret import interpret_call, interpret_doc, learned_something, release_events
from miramace.models import (
    CHICAGO,
    PCP_PARTY_ID,
    CallResult,
    CallTask,
    CaseState,
    CaseStatus,
    ComposeRequest,
    CostInfo,
    EscalationStatus,
    EscalationTrigger,
    InboundDoc,
    PatientProfile,
    SlotSpec,
    Supplier,
    SupplierStatus,
    Task,
    TaskKind,
    TaskStatus,
    Utterance,
    load_suppliers,
)
from miramace.notify import render_template, should_notify
from miramace.ports import Ports
from miramace.reducer import IllegalTransition
from miramace.snapshot import build_snapshot
from miramace.store import Store

ET = EventType
ITEM = "K0001 standard manual wheelchair (rental)"
RETRY_BH = 2.0

# ------------------------------------------------------------------------------------------ slots


def _s(id_: str, ask: str, type_: str = "bool", required: bool = True) -> SlotSpec:
    return SlotSpec(id=id_, ask=ask, type=type_, required=required)  # type: ignore[arg-type]


PROMISED_BY = _s("promised_by", "By when can we expect it?", "str")

# intent family -> slots. Persona facts are keyed by these slot ids (see scenarios/README.md).
SLOTS: dict[str, list[SlotSpec]] = {
    "request_order": [
        _s(
            "promised_action",
            "Can your office fax us Dr. Chen's visit note and a signed standard written "
            "order for a K0001 standard manual wheelchair?",
            "str",
        ),
        PROMISED_BY,
        _s("named_owner", "Who will be handling it?", "str", required=False),
    ],
    "followup_order": [
        _s("request_received", "Did you receive our request for her wheelchair paperwork?"),
        _s("order_status", "What is the status of the paperwork?", "str", required=False),
        _s("promised_by", "When will you fax it to us?", "str", required=False),
        _s("named_owner", "Who is handling it?", "str", required=False),
    ],
    "request_addendum": [
        _s(
            "will_send_addendum",
            "Could Dr. Chen fax an addendum to the visit note documenting the items "
            "Medicare requires that are missing?",
        ),
        PROMISED_BY,
    ],
    "request_correction": [
        _s(
            "will_send_correction",
            "Could Dr. Chen fax a corrected and signed order that includes the missing fields?",
        ),
        PROMISED_BY,
    ],
    "send_order": [
        _s("will_fax_order", "Could you fax the signed order and the visit note to the equipment supplier?"),
        _s("promised_by", "When will it go out?", "str", required=False),
    ],
    "screen": [
        _s("taking_new_medicare", "Are you taking new Medicare Part B patients?"),
        _s("k0001_in_stock", "Do you have a K0001 standard manual wheelchair in stock?"),
        _s("delivers_to_zip", "Do you deliver to ZIP code {zip}?"),
        _s("accepts_assignment_all_months", "Do you accept Medicare assignment for every rental month?"),
        _s("earliest_delivery_days", "How many days until you could deliver?", "int", required=False),
        _s("contact_name", "Who am I speaking with?", "str", required=False),
    ],
    "commit": [
        _s("reserve_unit", "Can you hold a K0001 wheelchair for her while the signed order is sent?"),
        _s("accepts_assignment_all_months", "Do you accept Medicare assignment for every rental month?"),
        _s("reference_number", "What reference number should we use?", "str", required=False),
    ],
    "verify_order_received": [
        _s("order_received", "Did you receive the signed order for her from Dr. Chen's office?"),
    ],
    "reverify": [
        _s("k0001_in_stock", "Do you still have a K0001 standard manual wheelchair available for her?"),
        _s("accepts_assignment_all_months", "Do you accept Medicare assignment for every rental month?"),
    ],
    "propose_slot": [
        _s("delivery_slot", "What is the earliest day and time you can deliver to her home?", "str"),
    ],
    "verify_delivered": [
        _s("delivered", "Was the wheelchair delivered to her home?"),
    ],
}


def slot_family(intent: str) -> str:
    if intent.startswith("followup_order_"):
        return "followup_order"
    if intent.startswith("send_order_"):
        return "send_order"
    return intent


PURPOSES: dict[str, str] = {
    "request_order": "Ask Dr. Chen's office to fax the visit note and a signed order for a K0001 wheelchair.",
    "followup_order": "Check whether Dr. Chen's office received our request and when the paperwork comes.",
    "request_addendum": "Ask Dr. Chen's office for an addendum documenting the missing Medicare criteria.",
    "request_correction": "Ask Dr. Chen's office for a corrected, complete signed order.",
    "send_order": "Ask Dr. Chen's office to fax the signed order to the committed equipment supplier.",
    "screen": "Check whether this supplier can take a new Medicare Part B K0001 wheelchair rental.",
    "commit": "Ask the supplier to hold a K0001 wheelchair for the patient and confirm assignment.",
    "verify_order_received": "Confirm the supplier received the signed order from Dr. Chen's office.",
    "reverify": "Re-confirm stock and assignment before scheduling delivery.",
    "propose_slot": "Ask the supplier for the earliest delivery time.",
    "verify_delivered": "Confirm the wheelchair was delivered.",
}

PHI_SUPPLIER_STATUSES = {SupplierStatus.QUALIFIED, SupplierStatus.COMMITTED}


@lru_cache(maxsize=1)
def _directory() -> dict[str, Supplier]:
    return {s.sid: s for s in load_suppliers()}


def supplier_name(sid: str | None) -> str:
    sup = _directory().get(sid or "")
    return sup.name if sup else (sid or "the supplier")


def phi_allowed(task: Task, state: CaseState) -> bool:
    """The PHI gate. PCP: yes. Supplier: never on screen; otherwise only when QUALIFIED/COMMITTED."""
    if task.kind == TaskKind.CALL_PCP:
        return True
    if task.intent == "screen":
        return False
    rec = state.suppliers.get(task.party_id)
    return rec is not None and rec.status in PHI_SUPPLIER_STATUSES


def build_call_task(task: Task, state: CaseState) -> CallTask:
    fam = slot_family(task.intent)
    zip_ = state.patient_info.address.zip if state.patient_info and state.patient_info.address else ""
    slots = [
        s.model_copy(update={"ask": s.ask.format(zip=zip_ or "the patient's ZIP")})
        for s in SLOTS.get(fam, [])
    ]
    phi = state.patient_info.phi() if (state.patient_info and phi_allowed(task, state)) else None
    purpose = PURPOSES.get(fam, task.intent.replace("_", " "))
    ctx: dict[str, str] = {"item": ITEM, "payer": "Original Medicare Part B"}
    if task.kind == TaskKind.CALL_PCP:
        pcp = state.pcp
        org, phone = (pcp.practice, pcp.phone) if pcp else ("Sunrise Family Medicine", "")
        party = "pcp"
        if pcp:
            ctx["prescriber"] = pcp.name
        if fam == "send_order":
            to = task.args.get("to_sid") or task.intent.removeprefix("send_order_")
            ctx["send_to"] = f"{supplier_name(to)} ({_directory()[to].phone})" if to in _directory() else to
            purpose = f"Ask Dr. Chen's office to fax the signed order to {supplier_name(to)}."
        if fam == "request_addendum" and task.args.get("wording"):
            ctx["missing"] = "; ".join(task.args["wording"])
        if fam == "request_correction" and task.args.get("fields"):
            ctx["missing_fields"] = ", ".join(task.args["fields"])
    else:
        sup = _directory().get(task.party_id)
        org, phone = (sup.name, sup.phone) if sup else (task.party_id, "")
        party = "supplier"
        ctx["patient_zip"] = zip_
        rec = state.suppliers.get(task.party_id)
        if rec and rec.ref:
            ctx["reference"] = rec.ref
    return CallTask(
        task_id=task.task_id,
        party=party,
        party_id=task.party_id,
        org_name=org,
        phone=phone,
        intent=task.intent,
        purpose=purpose,
        slots=slots,
        phi=phi,
        context=ctx,
    )


def _failed_result(ct: CallTask, exc: BaseException) -> CallResult:
    return CallResult(
        task_id=ct.task_id,
        outcome="FAILED",
        transcript=[
            Utterance(
                turn=0, speaker="system", name="system", text=f"Call failed: {type(exc).__name__}", ts=0.0
            )
        ],
        claims=[],
        flags=[f"exception:{type(exc).__name__}"],
        duration_s=0,
        callback_after=None,
        transport="fake",
    )


class _IntentStore:
    """IntentAPI store adapter: reads the real store, appends through the executor (observers + on_event)."""

    def __init__(self, ex: Executor) -> None:
        self.ex = ex

    def state(self, case_id: str) -> CaseState:
        return self.ex.store.state(case_id)

    def append(self, event: Event) -> Event:
        out = self.ex.append([event], strict=True)
        return out[0]


class Executor:
    def __init__(
        self, ports: Ports, store: Store, clock: Any, on_event: Callable[[Event], None] | None = None
    ) -> None:
        self.ports, self.store, self.clock, self.on_event = ports, store, clock, on_event
        self.observers = [
            p for p in (ports.inbound, ports.caller_pcp, ports.caller_supplier) if hasattr(p, "observe")
        ]
        self.history: dict[str, list[Utterance]] = {}  # case_id -> patient chat history

    # ---------------------------------------------------------------------------------- append
    def observe(self, e: Event) -> None:
        for o in self.observers:
            o.observe(e)

    def append(self, events: list[Event], *, strict: bool = False) -> list[Event]:
        out: list[Event] = []
        for ev in events:
            try:
                stored = self.store.append(ev)
            except IllegalTransition as exc:
                if strict:
                    raise
                ref = ev.source.ref
                audit = make_event(
                    ev.case_id,
                    ET.EXTRACTION_FAILED,
                    {"task_id": ref, "field": f"event:{ev.type.value}", "reason": str(exc)},
                    ev.source,
                    ev.sim_time,
                )
                stored = self.store.append(audit)
            out.append(stored)
            self.observe(stored)
            if self.on_event:
                self.on_event(stored)
        if out:
            try:
                self.store.sync_projections(out[0].case_id, self.store.state(out[0].case_id))
            except Exception:
                pass
        return out

    def _reschedule(self, task: Task, due: datetime, attempts: int) -> None:
        # Store has no reschedule API (B1); a PENDING task's due_at is ours to move. Times stored in UTC.
        from datetime import UTC

        self.store._db.execute(
            "UPDATE tasks SET due_at=?, attempts=? WHERE task_id=?",
            (due.astimezone(UTC).isoformat(), attempts, task.task_id),
        )

    # ---------------------------------------------------------------------------------- execute
    async def execute(self, task: Task, state: CaseState) -> list[Event]:
        if task.kind == TaskKind.NOTIFY_PATIENT:
            return await self._notify(task, state)
        return await self._call(task, state)

    async def _call(self, task: Task, state: CaseState) -> list[Event]:
        ct = build_call_task(task, state)
        port = self.ports.caller_pcp if task.kind == TaskKind.CALL_PCP else self.ports.caller_supplier
        try:
            result = await port.run_call(ct)
        except Exception as exc:  # a broken transport must not crash the case
            result = _failed_result(ct, exc)
        self.clock.advance(max(0, result.duration_s))
        now = self.clock.now()
        tid = self.store.save_transcript(state.case_id, task.task_id, task.party_id, result, now)
        src = Source(kind="TRANSCRIPT", ref=tid)
        out = self.append(
            [
                make_event(
                    state.case_id,
                    ET.CALL_COMPLETED,
                    {"task_id": task.task_id, "transcript_id": tid, "outcome": result.outcome},
                    src,
                    now,
                )
            ]
        )
        fresh = self.store.state(state.case_id)
        domain = interpret_call(task, result, fresh, now, transcript_id=tid)
        out += self.append(domain)
        attempts = task.attempts + 1
        planner_owned = task.intent == "screen" or task.intent.startswith("followup_order_")
        done = (
            planner_owned
            or (result.outcome == "COMPLETED" and learned_something(domain))
            or result.outcome == "REFUSED"
        )
        if done:
            self.store.mark_task(task.task_id, TaskStatus.DONE, attempts=attempts, last_result=result.outcome)
        elif attempts < task.max_attempts:
            due = result.callback_after or add_business_hours(now, RETRY_BH)
            self._reschedule(task, max(due, now), attempts)
        else:
            self.store.mark_task(
                task.task_id, TaskStatus.FAILED, attempts=attempts, last_result=result.outcome
            )
            out += self._exhausted(task, result, src, now)
        return out

    def _exhausted(self, task: Task, result: CallResult, src: Source, now: datetime) -> list[Event]:
        state = self.store.state(task.case_id)
        if task.kind == TaskKind.CALL_SUPPLIER:
            if task.intent in ("verify_order_received", "reverify"):
                return []  # planner releases after repeated SUPPLIER_NO_ANSWER (F1)
            return self.append(
                release_events(
                    state,
                    task.party_id,
                    f"{task.intent} failed {task.attempts + 1}x ({result.outcome})",
                    src,
                    now,
                )
            )
        if any(
            e.trigger == EscalationTrigger.ORDER_STALLED and e.status == EscalationStatus.OPEN
            for e in state.escalations.values()
        ):
            return []
        ev = open_escalation(
            EscalationTrigger.ORDER_STALLED,
            state,
            now,
            {
                "what": task.intent.replace("_", " "),
                "followups": task.attempts + 1,
                "key": f"task:{task.idem_key}",
            },
            source=src,
        )
        return self.append([ev])

    # ---------------------------------------------------------------------------------- notify
    async def _notify(self, task: Task, state: CaseState) -> list[Event]:
        case_id, now = state.case_id, self.clock.now()
        purpose = task.args.get("purpose") or task.intent
        profile = state.patient_info.profile if state.patient_info else None
        if state.case.state != CaseStatus.ACTIVE:
            self.store.mark_task(task.task_id, TaskStatus.CANCELLED)
            return []
        if not should_notify(state, purpose, now):
            nxt = (
                next_patient_time(now, profile)
                if not is_patient_time(now, profile)
                else next_patient_time((now + timedelta(days=1)).replace(hour=0, minute=0), profile)
            )
            self._reschedule(task, nxt, task.attempts)
            return []
        snap = build_snapshot(state, now)
        try:
            text = await self.ports.companion.compose(
                ComposeRequest(
                    case_id=case_id,
                    purpose=purpose,
                    snapshot=snap,
                    profile=profile or PatientProfile(),
                    history=self.history.get(case_id, [])[-10:],
                )
            )
        except Exception:
            text = ""
        if not text or grounding_check(text, snap, profile):
            text = render_template(purpose, snap)
        grounded = not grounding_check(text, snap, profile)
        if not grounded:  # never send an ungrounded message
            self.store.mark_task(
                task.task_id, TaskStatus.FAILED, attempts=task.attempts + 1, last_result="grounding_hold"
            )
            return []
        reply = await self.ports.patient.deliver(case_id, purpose, text, now)
        src = Source(kind="SYSTEM", ref=f"notify:{task.idem_key}")
        evs = [
            make_event(
                case_id,
                ET.PATIENT_NOTIFIED,
                {"purpose": purpose, "text": text, "grounding_ok": True, "answered": reply is not None},
                src,
                now,
            )
        ]
        if purpose == "cost_disclosure":
            evs.append(make_event(case_id, ET.COST_DISCLOSED, {"cost": CostInfo()}, src, now))
        out = self.append(evs)
        hist = self.history.setdefault(case_id, [])
        hist.append(Utterance(turn=len(hist), speaker="caller", name="Mira Mace", text=text, ts=0.0))
        if reply is not None:
            hist.append(
                Utterance(turn=len(hist), speaker="counterparty", name="Eleanor", text=reply.text, ts=0.0)
            )
            intents = list(reply.intents)
            if not intents and reply.text:
                snap2 = build_snapshot(self.store.state(case_id), now)
                try:
                    intents = (await self.ports.companion.respond(snap2, hist[:-1], reply.text)).intents
                except Exception:
                    intents = []
            api = IntentAPI(_IntentStore(self), self.clock)
            before = len(self.store.events(case_id))
            for intent in intents:
                api.submit(case_id, intent)
            out += self.store.events(case_id)[before:]
        attempts = task.attempts + 1
        ladder = purpose in ("welcome_intake", "cost_disclosure", "delivery_slot")
        if reply is None and not ladder and attempts < task.max_attempts:
            self._reschedule(
                task,
                next_patient_time(
                    (now + timedelta(days=1)).astimezone(CHICAGO).replace(hour=0, minute=0), profile
                ),
                attempts,
            )
        else:
            self.store.mark_task(
                task.task_id,
                TaskStatus.DONE,
                attempts=attempts,
                last_result="answered" if reply else "unanswered",
            )
        return out

    # ---------------------------------------------------------------------------------- docs
    async def ingest_doc(self, doc: InboundDoc, state: CaseState) -> list[Event]:
        try:
            claims = await self.ports.doc_reader.read_doc(doc)
        except Exception:
            claims = []
        return self.append(interpret_doc(doc, claims, state, self.clock.now()))


__all__ = ["Executor", "SLOTS", "build_call_task", "phi_allowed", "slot_family", "PCP_PARTY_ID"]
