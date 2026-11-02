"""B1 — invariants: each trips on a hand-built bad state/history and passes on a good one."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from miramace.clock import SIM_START
from miramace.events import EventType, Source, make_event
from miramace.invariants import (
    InvariantViolation,
    assert_idempotent_outbound,
    assert_liveness,
    assert_patient_contact_ok,
    assert_schedule_preconditions,
    assert_single_commit,
    assert_traceable,
    check_all,
    schedule_blockers,
)
from miramace.models import (
    CHICAGO,
    CaseStatus,
    DeliveryStatus,
    PatientProfile,
    SupplierStatus,
    Task,
    TaskKind,
    TaskStatus,
    idem_key,
)
from test_reducer import CASE, A, B, happy_path, opened, ready_to_schedule

ET = EventType


def task(intent="request_order", version=1, status=TaskStatus.PENDING, key=None) -> Task:
    return Task(
        task_id=f"t-{intent}-{version}",
        case_id=CASE,
        kind=TaskKind.CALL_PCP,
        party_id="pcp",
        intent=intent,
        version=version,
        idem_key=key or idem_key(CASE, "pcp", intent, version),
        due_at=SIM_START,
        status=status,
    )


def test_single_commit():
    st = happy_path().state()
    assert_single_commit(st)
    bad = st.model_copy(deep=True)
    bad.suppliers[B].status = SupplierStatus.COMMITTED
    with pytest.raises(InvariantViolation):
        assert_single_commit(bad)


def test_schedule_preconditions():
    log = happy_path()
    st = log.state()
    assert_schedule_preconditions(st, log.events)
    for breaker in ("received", "cost", "lcd", "assignment"):
        bad = st.model_copy(deep=True)
        if breaker == "received":
            bad.order_received_by = B
        elif breaker == "cost":
            bad.patient_flags.discard("cost_accepted")
        elif breaker == "lcd":
            bad.lcd = None
        else:
            bad.assignment_by_sid.pop(A)
        with pytest.raises(InvariantViolation):
            assert_schedule_preconditions(bad, log.events)
    # stale verification relative to the DELIVERY_SCHEDULED event time
    bad = st.model_copy(deep=True)
    bad.suppliers[A].verified_at = SIM_START - timedelta(days=3)
    with pytest.raises(InvariantViolation):
        assert_schedule_preconditions(bad, log.events)
    # not scheduled -> nothing to check
    assert_schedule_preconditions(opened().state())


def test_schedule_blockers_predicate():
    log = ready_to_schedule()
    st = log.state()
    assert schedule_blockers(st, A, log.t) == []
    assert schedule_blockers(st, B, log.t)
    assert schedule_blockers(st, A, log.t + timedelta(hours=30))


def test_liveness():
    st = opened().state()
    assert_liveness(st, [task()], pending_inbound=False)
    assert_liveness(st, [], pending_inbound=True)
    with pytest.raises(InvariantViolation):
        assert_liveness(st, [task(status=TaskStatus.DONE)], pending_inbound=False)
    done = happy_path().state()
    assert done.case.state == CaseStatus.DONE
    assert_liveness(done, [task(status=TaskStatus.DONE)], pending_inbound=False)
    with pytest.raises(InvariantViolation):
        assert_liveness(done, [task()], pending_inbound=False)
    paused = opened().add(ET.CASE_PAUSED, {"reason": "hospital"}).state()
    assert_liveness(paused, [], pending_inbound=False)


def test_idempotent_outbound():
    good = opened()
    good.add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    good.add(ET.ORDER_REQUEST_VERIFIED, {"received": False})
    good.add(ET.ORDER_REQUESTED, {"request_ref": "r2", "version": 2})
    assert_idempotent_outbound([task(), task(version=2)], good.events)
    with pytest.raises(InvariantViolation):  # duplicate idem key
        assert_idempotent_outbound([task(), task(intent="other", key=task().idem_key)], good.events)
    bad = opened()
    bad.add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    bad.add(ET.ORDER_REQUEST_VERIFIED, {"received": True})
    bad.add(ET.ORDER_REQUESTED, {"request_ref": "r2", "version": 2})
    with pytest.raises(InvariantViolation):  # resend without a not-received verify
        assert_idempotent_outbound([], bad.events)
    v1 = opened().add(ET.ORDER_REQUESTED, {"request_ref": "r", "version": 1})
    with pytest.raises(InvariantViolation):  # planned resend task without verify
        assert_idempotent_outbound([task(version=2)], v1.events)


def test_traceable():
    log = opened().add(ET.CONSENT_GIVEN, {"scope": "x"}, kind="TRANSCRIPT", ref="tr-1")
    log.add(ET.CALL_COMPLETED, {"task_id": "t", "transcript_id": "tr-1", "outcome": "COMPLETED"})
    assert_traceable(log.events, {"tr-1"})
    with pytest.raises(InvariantViolation):
        assert_traceable(log.events, set())
    empty_ref = make_event(CASE, ET.CONSENT_GIVEN, {"scope": "x"}, Source(kind="RULE", ref=""), SIM_START)
    with pytest.raises(InvariantViolation):
        assert_traceable([empty_ref], set())


def _at(h: int, m: int = 0, day: int = 2) -> datetime:
    return datetime(2026, 11, day, h, m, tzinfo=CHICAGO)


def test_patient_contact_ok():
    def notified(t, grounded=True):
        return make_event(
            CASE,
            ET.PATIENT_NOTIFIED,
            {"purpose": "status_update", "text": "x", "grounding_ok": grounded, "answered": True},
            Source(kind="RULE", ref="notify"),
            t,
        )

    def call(t):
        return make_event(
            CASE,
            ET.CALL_COMPLETED,
            {"task_id": "t", "transcript_id": "tr", "outcome": "COMPLETED"},
            Source(kind="TRANSCRIPT", ref="tr"),
            t,
        )

    prof = PatientProfile(best_hours=(10, 18))
    assert_patient_contact_ok([notified(_at(11)), call(_at(16, 50)), call(_at(17, 10))], prof)
    for bad in (notified(_at(11), grounded=False), notified(_at(19)), call(_at(8)), call(_at(12, day=7))):
        with pytest.raises(InvariantViolation):
            assert_patient_contact_ok([bad], prof)


def test_check_all_happy_path():
    log = happy_path()
    st = log.state()
    assert st.delivery.state == DeliveryStatus.POD_CONFIRMED
    check_all(
        st, [task(status=TaskStatus.DONE)], log.events, now=log.t, transcript_ids=set(), pending_inbound=False
    )
    # pre-plan check at loop start: no tasks yet, liveness skipped
    st0 = opened().state()
    check_all(st0, [], [], now=SIM_START, transcript_ids=set(), pending_inbound=False, check_liveness=False)
    with pytest.raises(InvariantViolation):
        check_all(st0, [], [], now=SIM_START, transcript_ids=set(), pending_inbound=False)
    assert isinstance(InvariantViolation("x"), AssertionError)
