"""B6 Engine — the run loop (plan-core "Clock & run loop") on a virtual clock.

    loop:
      state = reduce(events); invariants.check_all(check_liveness=False)
      dec = plan(state, now); if dec.events: append (source RULE) and re-plan
      reconcile_tasks(dec.tasks); invariants.check_all(check_liveness=True)
      next = min(next pending task due, next inbound arrival); none -> stop
      clock.advance_to(next); inbound docs first (executor.ingest_doc) else executor.execute(task)

Stops with: DONE / CANCELLED (terminal case), BLOCKED_ON_HUMAN (an escalation is open and nothing but
lease keep-alive work remains), STUCK (no work and no open escalation, or an invariant violation — recorded,
not raised, so the CLI can show it), MAX_STEPS (step budget or `until` reached).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from miramace import escalation, invariants
from miramace.clock import VirtualClock
from miramace.events import Event, EventType
from miramace.executor import Executor
from miramace.llm.port import cost_usd
from miramace.models import CaseState, CaseStatus, EscalationStatus, Task
from miramace.planner import Policy, plan
from miramace.ports import Ports
from miramace.store import Store

ET = EventType
Status = Literal["DONE", "BLOCKED_ON_HUMAN", "STUCK", "MAX_STEPS", "CANCELLED"]


class RunSummary(BaseModel):
    case_id: str
    scenario: str | None = None
    status: Status
    sim_start: datetime
    sim_end: datetime
    steps: int
    calls: int
    escalations: list[str] = Field(default_factory=list)  # trigger names opened (in order)
    open_escalations: list[str] = Field(default_factory=list)  # eids
    invariant_violations: list[str] = Field(default_factory=list)
    usage_tokens: int = 0
    cost_usd: float | None = None


def _open(state: CaseState) -> list[str]:
    return [e.eid for e in state.escalations.values() if e.status == EscalationStatus.OPEN]


def _keepalive(task: Task) -> bool:
    return task.intent == "reverify" and task.args.get("reason") == "lease"


def _check(state: CaseState, store: Store, ports: Ports, now: datetime, live: bool, viol: list[str]) -> bool:
    try:
        invariants.check_all(
            state,
            store.tasks(state.case_id),
            store.events(state.case_id),
            now=now,
            transcript_ids=store.transcript_ids(state.case_id),
            pending_inbound=ports.inbound.next_arrival(state.case_id) is not None,
            check_liveness=live,
        )
        return True
    except invariants.InvariantViolation as exc:
        viol.append(str(exc))
        return False


def summarize(
    case_id: str, store: Store, ports: Ports, status: Status, steps: int, viol: list[str], end: datetime
) -> RunSummary:
    events = store.events(case_id)
    state = store.state(case_id)
    case = store.case(case_id) or {}
    usage = 0
    cost: float | None = None
    led = getattr(ports.llm, "ledger", None) if ports.llm is not None else None
    if isinstance(led, list):
        usage = sum(c.in_tok + c.out_tok for c in led)
        cost = cost_usd(led)
    else:
        for tr in store.transcripts(case_id):
            usage += sum(c.in_tok + c.out_tok for c in tr.result.usage)
    return RunSummary(
        case_id=case_id,
        scenario=case.get("scenario"),
        status=status,
        sim_start=events[0].sim_time if events else end,
        sim_end=end,
        steps=steps,
        calls=sum(1 for e in events if e.type == ET.CALL_COMPLETED),
        escalations=[str(e.trigger) for e in state.escalations.values()],
        open_escalations=_open(state),
        invariant_violations=viol,
        usage_tokens=usage,
        cost_usd=cost,
    )


async def run_case(
    case_id: str,
    ports: Ports,
    store: Store,
    clock: VirtualClock,
    policy: Policy = Policy(),  # noqa: B008
    *,
    max_steps: int = 400,
    until: datetime | None = None,
    on_event: Callable[[Event], None] | None = None,
    executor: Executor | None = None,
) -> RunSummary:
    ex = executor or Executor(ports, store, clock, on_event=on_event)
    for e in store.events(case_id):  # prime simulators (observe is idempotent per event_id)
        ex.observe(e)
    viol: list[str] = []
    steps = 0
    status: Status = "MAX_STEPS"
    while True:
        state = store.state(case_id)
        now = clock.now()
        if not _check(state, store, ports, now, False, viol):
            status = "STUCK"
            break
        dec = plan(state, now, policy)
        if dec.events:
            ex.append(dec.events)
            steps += 1
            if steps >= max_steps:
                break
            continue
        store.reconcile_tasks(case_id, dec.tasks, now)
        cs = CaseStatus(state.case.state)
        if cs != CaseStatus.ACTIVE:
            status = (
                "DONE"
                if cs == CaseStatus.DONE
                else ("BLOCKED_ON_HUMAN" if cs == CaseStatus.PAUSED else "CANCELLED")
            )
            _check(state, store, ports, now, True, viol)
            break
        task = store.next_pending_due(case_id)
        t_in = ports.inbound.next_arrival(case_id)
        open_esc = _open(state)
        if task is None and t_in is None:
            if open_esc:
                status = "BLOCKED_ON_HUMAN"
            else:
                _check(state, store, ports, now, True, viol)
                status = "STUCK"
            break
        if not _check(state, store, ports, now, True, viol):
            status = "STUCK"
            break
        use_doc = t_in is not None and (task is None or t_in <= task.due_at)
        nxt = t_in if use_doc else task.due_at  # type: ignore[union-attr]
        if open_esc and not use_doc and task is not None and _keepalive(task):
            status = "BLOCKED_ON_HUMAN"
            break
        if until is not None and nxt > until:
            break
        clock.advance_to(max(now, nxt))
        if use_doc:
            for doc in ports.inbound.pop_due(case_id, clock.now()):
                await ex.ingest_doc(doc, store.state(case_id))
        else:
            assert task is not None
            await ex.execute(task, store.state(case_id))
        steps += 1
        if steps >= max_steps:
            break
    return summarize(case_id, store, ports, status, steps, viol, clock.now())


async def resume_after_resolution(
    case_id: str,
    eid: str,
    option_id: str,
    note: str | None,
    resolver: str,
    ports: Ports,
    store: Store,
    clock: VirtualClock,
    policy: Policy = Policy(),  # noqa: B008
    **kw: Any,
) -> RunSummary:
    """Append HUMAN_RESOLVED + effect events (escalation.resolve; raises ValueError on a bad eid/option),
    then continue the run loop."""
    ex = Executor(ports, store, clock, on_event=kw.get("on_event"))
    for e in store.events(case_id):
        ex.observe(e)
    events = escalation.resolve(store.state(case_id), eid, option_id, note, resolver, clock.now())
    ex.append(events, strict=True)
    return await run_case(case_id, ports, store, clock, policy, executor=ex, **kw)
