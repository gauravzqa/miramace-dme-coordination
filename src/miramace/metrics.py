"""B6 Engine — scenario scoring and aggregate metrics.

- autonomous_close: DONE with 0 escalations.
- false_close (hard gate): the case reached DELIVERY_SCHEDULED or later while the scenario's ground truth
  (oracle on the fact sheet, not on extracted facts) says the claim would NOT pass.
- escalation tp/fp/fn: opened trigger names vs expect.escalations (sets).
- phi_before_qualification: caller utterances containing the patient's name/DOB/MBI in calls to a supplier
  that was not QUALIFIED/COMMITTED at call time (state replayed up to the CALL_COMPLETED event).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from miramace.agents.prompts import contains_phi
from miramace.events import EventType
from miramace.models import PCP_PARTY_ID, CaseState, SupplierStatus
from miramace.reducer import reduce_all
from miramace.store import Store

ET = EventType


class ScenarioScore(BaseModel):
    name: str
    status: str
    expected_status: str
    autonomous_close: bool
    false_close: bool
    escalations_expected: list[str] = Field(default_factory=list)
    escalations_opened: list[str] = Field(default_factory=list)
    escalation_tp: int = 0
    escalation_fp: int = 0
    escalation_fn: int = 0
    sim_days: float = 0.0
    calls: int = 0
    invariant_violations: int = 0
    phi_before_qualification: int = 0
    passed: bool = False


class Metrics(BaseModel):
    scenarios: int
    autonomous_closes: int
    false_closes: int
    escalation_precision: float
    escalation_recall: float
    invariant_violations: int
    phi_leaks: int
    all_passed: bool


def phi_before_qualification(store: Store, case_id: str) -> int:
    events = store.events(case_id)
    if not events:
        return 0
    patient = events[0].parsed().patient  # type: ignore[attr-defined]
    phi = patient.phi()
    leaks = 0
    for i, e in enumerate(events):
        if e.type != ET.CALL_COMPLETED:
            continue
        tr = store.transcript(e.payload["transcript_id"])
        if tr is None or tr.party_id == PCP_PARTY_ID:
            continue
        before = reduce_all(events[:i]) if i else None
        rec = before.suppliers.get(tr.party_id) if before else None
        ok = rec is not None and rec.status in (SupplierStatus.QUALIFIED, SupplierStatus.COMMITTED)
        if ok:
            continue
        leaks += sum(1 for u in tr.result.transcript if u.speaker == "caller" and contains_phi(u.text, phi))
    return leaks


def score(summary: Any, scenario: Any, state: CaseState, store: Store) -> ScenarioScore:
    events = store.events(summary.case_id)
    scheduled = any(e.type == ET.DELIVERY_SCHEDULED for e in events)
    false_close = scheduled and not scenario.ground_truth.claim_would_pass
    expected = set(scenario.expect.escalations)
    opened = list(summary.escalations)
    tp = len(expected & set(opened))
    fp = len(set(opened) - expected)
    fn = len(expected - set(opened))
    days = (summary.sim_end - summary.sim_start).total_seconds() / 86400
    phi = phi_before_qualification(store, summary.case_id)
    nviol = len(summary.invariant_violations)
    within = scenario.expect.max_sim_days is None or days <= scenario.expect.max_sim_days
    passed = (
        summary.status == scenario.expect.final
        and not false_close
        and fp == 0
        and fn == 0
        and nviol == 0
        and phi == 0
        and within
    )
    del state
    return ScenarioScore(
        name=scenario.name,
        status=summary.status,
        expected_status=scenario.expect.final,
        autonomous_close=summary.status == "DONE" and not opened,
        false_close=false_close,
        escalations_expected=sorted(expected),
        escalations_opened=opened,
        escalation_tp=tp,
        escalation_fp=fp,
        escalation_fn=fn,
        sim_days=round(days, 2),
        calls=summary.calls,
        invariant_violations=nviol,
        phi_before_qualification=phi,
        passed=passed,
    )


def aggregate(scores: list[ScenarioScore]) -> Metrics:
    tp = sum(s.escalation_tp for s in scores)
    fp = sum(s.escalation_fp for s in scores)
    fn = sum(s.escalation_fn for s in scores)
    return Metrics(
        scenarios=len(scores),
        autonomous_closes=sum(s.autonomous_close for s in scores),
        false_closes=sum(s.false_close for s in scores),
        escalation_precision=tp / (tp + fp) if tp + fp else 1.0,
        escalation_recall=tp / (tp + fn) if tp + fn else 1.0,
        invariant_violations=sum(s.invariant_violations for s in scores),
        phi_leaks=sum(s.phi_before_qualification for s in scores),
        all_passed=bool(scores) and all(s.passed for s in scores),
    )
