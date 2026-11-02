"""Verifier: walk plan-core flows through reducer + planner and flag STUCK states / defects."""

from __future__ import annotations

from datetime import timedelta

import pytest
from _v1_driver import ADDR, ALL_MET, ET, GOOD, H, SWO_OK, Sim, consented_sim, cost, intents, keys, note

from miramace.clock import add_business_hours
from miramace.escalation import resolve
from miramace.models import DeliveryStatus, OrderStatus, PatientStatus
from miramace.reducer import IllegalTransition, reduce
from miramace.events import make_event, Source


def _ranked(s):
    return s.state.ranked


def happy_until_received() -> tuple[Sim, str, str]:
    s = consented_sim()
    a, b = _ranked(s)[0], _ranked(s)[1]
    s.add(ET.SUPPLIER_QUALIFIED, {"sid": a, "facts": GOOD, "verified_at": s.now})
    s.add(ET.SUPPLIER_QUALIFIED, {"sid": b, "facts": GOOD, "verified_at": s.now})
    s.add(ET.SUPPLIER_COMMITTED, {"sid": a, "ref": "r1", "expires_at": s.now + 72 * H})
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    s.add(ET.COMMITMENT_MADE, {"cid": "pcp-1", "party": "pcp", "what": "fax", "due_at": s.now + 24 * H})
    s.add(ET.COST_DISCLOSED, cost())
    s.add(ET.COST_ACCEPTED)
    s.at(s.now + 4 * H)
    s.add(ET.VISIT_NOTE_RECEIVED, note("n1"))
    s.add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
    s.add(ET.SWO_SIGNED_RECEIVED, {"swo": SWO_OK})
    s.add(ET.SWO_EVALUATED, {"complete": True, "missing": [], "hcpcs": "K0001"})
    s.apply_planned_events()  # ORDER_VALIDATED
    assert s.state.order.state == OrderStatus.VALID
    s.add(ET.ORDER_SENT, {"to_sid": a})
    s.add(ET.ORDER_RECEIVED, {"by_sid": a})
    return s, a, b


def test_v1_happy_path_reaches_case_done():
    """Sanity: the straight-line path does not get STUCK."""
    s, a, _ = happy_until_received()
    dec = s.plan()
    assert "propose_slot" in intents(dec)
    slot = add_business_hours(s.now, 8)
    s.add(ET.DELIVERY_SLOT_PROPOSED, {"sid": a, "slot": slot})
    assert "delivery_slot" in intents(s.plan())
    s.add(ET.DELIVERY_SCHEDULED, {"sid": a, "slot": slot})
    assert "verify_delivered" in intents(s.plan())
    s.add(ET.DELIVERED, {"sid": a, "at": slot})
    assert "delivered_checkin" in intents(s.plan())
    s.add(ET.POD_CONFIRMED)
    assert [e.type for e in s.plan().events] == [ET.CASE_DONE]


def test_v1_patient_declines_proposed_slot_is_stuck():
    """DEFECT: patient answers the delivery_slot notify but does not accept (wants another time).
    `reschedule` is rejected in SCHEDULING (intents.py), propose_slot is not re-planned while a slot exists,
    and the delivery_slot notify version (10*stamp + notify_attempts + 1) is identical to the one already
    DONE because notify_attempts reset to 0 on answer -> no new task, no event -> STUCK."""
    s, a, _ = happy_until_received()
    slot = add_business_hours(s.now, 8)
    s.add(ET.DELIVERY_SLOT_PROPOSED, {"sid": a, "slot": slot})
    first = keys(s.plan())
    s.add(ET.PATIENT_NOTIFIED, {"purpose": "delivery_slot", "text": "x", "grounding_ok": True, "answered": True})
    s.at(s.now + 2 * H)
    after = keys(s.plan())
    new = after - first
    assert new, f"no new task after an answered-but-unaccepted slot offer; desired={after} (already DONE)"


def test_v1_cost_disclosed_but_unanswered_never_retried():
    """DEFECT: rule 8 only fires in CONSENTED. Once COST_DISCLOSED is recorded, an unanswered disclosure is
    never retried (no 3-try ladder / caregiver / PATIENT_UNREACHABLE) and patient.due_at is never consulted."""
    s = consented_sim()
    a = _ranked(s)[0]
    s.add(ET.SUPPLIER_QUALIFIED, {"sid": a, "facts": GOOD, "verified_at": s.now})
    s.add(ET.SUPPLIER_COMMITTED, {"sid": a, "ref": "r1", "expires_at": s.now + 72 * H})
    s.add(ET.COST_DISCLOSED, cost())
    s.add(ET.PATIENT_NOTIFIED, {"purpose": "cost_disclosure", "text": "x", "grounding_ok": True, "answered": False})
    s.at(s.now + 48 * H)
    dec = s.plan()
    assert s.state.patient.state == PatientStatus.COST_DISCLOSED
    assert "cost_disclosure" in intents(dec), f"no cost_disclosure retry planned: {intents(dec)}"


def test_v1_welcome_answered_with_address_only_never_reasked():
    """DEFECT (same root cause): patient replies to welcome with address but no consent -> INTAKE forever."""
    s = Sim()
    s.add(ET.PATIENT_NOTIFIED, {"purpose": "welcome_intake", "text": "hi", "grounding_ok": True, "answered": True})
    s.add(ET.ADDRESS_CONFIRMED, {"address": ADDR})
    before = {k for k in keys(Sim().plan()) if k[1] == "welcome_intake"}
    after = {k for k in keys(s.at(s.now + 24 * H).plan()) if k[1] == "welcome_intake"}
    assert s.state.patient.state == PatientStatus.INTAKE
    assert after - before, f"welcome never re-asked for consent; desired={after} == already-done {before}"


def test_v1_cost_concern_before_disclosure_cannot_be_resolved_with_suggested_option():
    """DEFECT: COST_CONCERN raised before COST_DISCLOSED (e.g. Eleanor says 'I can't afford this' at intake).
    The suggested option nurse_resolved_accepted emits COST_ACCEPTED, which the reducer rejects
    ('COST_ACCEPTED before COST_DISCLOSED'), so the whole resolve batch is illegal."""
    from miramace.escalation import open_escalation
    from miramace.models import EscalationTrigger

    s = consented_sim()
    s.add(ET.CONCERN_RAISED, {"kind": "cost", "utterance": "I can't afford this"})
    ev = open_escalation(EscalationTrigger.COST_CONCERN, s.state, s.now, {"utterance": "x"})
    s._apply(ev)
    eid = next(iter(s.state.escalations))
    out = resolve(s.state, eid, "nurse_resolved_accepted", None, "nurse", s.now)
    st = s.state
    for e in out:  # must apply cleanly
        st = reduce(st, e.model_copy(update={"seq": st.last_seq + 1}))


def test_v1_ghost_supplier_lease_expiry_has_no_wakeup_task():
    """DEFECT: ghost failover relies solely on rule 12 (lease end). 'reverify failed twice -> release' is not
    implemented, SUPPLIER_NO_ANSWER on a COMMITTED record is a reducer no-op, and no planned task is due at
    or after lease end. Once the lease-reverify task has FAILED (same idem key re-desired forever), the run
    loop has nothing to advance to and never reaches the time at which SUPPLIER_RELEASED would be emitted."""
    s = consented_sim()
    a, b = _ranked(s)[0], _ranked(s)[1]
    s.add(ET.SUPPLIER_QUALIFIED, {"sid": a, "facts": GOOD, "verified_at": s.now})
    s.add(ET.SUPPLIER_QUALIFIED, {"sid": b, "facts": GOOD, "verified_at": s.now})
    s.add(ET.SUPPLIER_COMMITTED, {"sid": a, "ref": "r1", "expires_at": s.now + 72 * H})
    end = s.now + 72 * H
    s.at(end - 2 * H)
    dec = s.plan()
    rev = [t for t in dec.tasks if t.intent == "reverify"]
    assert rev and rev[0].due_at < end
    # pretend the reverify task FAILED (no answer twice): planner re-desires the same key; nothing else is due
    later = [t for t in dec.tasks if t.due_at >= end and t.idem_key != rev[0].idem_key]
    sup_events = [e.type for e in dec.events]
    assert later or ET.SUPPLIER_RELEASED in sup_events, (
        "no task due at/after lease end and no release -> loop cannot wake up to release the ghost"
    )


def _docs_check_with_lcd_gap() -> Sim:
    s = consented_sim()
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    s.add(ET.COMMITMENT_MADE, {"cid": "pcp-1", "party": "pcp", "what": "fax", "due_at": s.now + 24 * H})
    s.add(ET.VISIT_NOTE_RECEIVED, note("n1"))
    gap = ALL_MET.model_copy(deep=True)
    from miramace.models import LCDCriterion

    gap.results[LCDCriterion.HOME_ACCOMMODATES] = "not_documented"
    s.add(ET.LCD_EVALUATED, {"checklist": gap, "missing": [LCDCriterion.HOME_ACCOMMODATES]})
    return s


def test_v1_lcd_addendum_escalates_immediately_if_signed_swo_arrives_first():
    """DEFECT: addendum requested (addendum_requests=1) and then the signed SWO arrives before the addendum
    note (ADDENDUM_REQUESTED -> SIGNED is legal). Rule 4 runs in SIGNED and, because addendum_requests>0 and
    the OLD checklist still has the gap, opens LCD_MISSING_AFTER_ADDENDUM without waiting for the addendum."""
    s = _docs_check_with_lcd_gap()
    assert "request_addendum" in intents(s.plan())
    from miramace.models import LCDCriterion

    s.add(ET.ADDENDUM_REQUESTED, {"missing": [LCDCriterion.HOME_ACCOMMODATES]})
    s.add(ET.SWO_SIGNED_RECEIVED, {"swo": SWO_OK})
    s.add(ET.SWO_EVALUATED, {"complete": True, "missing": [], "hcpcs": "K0001"})
    assert s.state.order.state == OrderStatus.SIGNED
    trig = [e.parsed().escalation.trigger for e in s.plan().events if e.type == ET.ESCALATION_OPENED]
    assert not trig, f"escalated {trig} before the requested addendum could arrive"


def test_v1_code_drift_proceed_manual_leaves_case_stuck():
    """DEFECT: CODE_DRIFT resolved with proceed_new_code_manual ('blocks' per plan-core) has no effect; the
    reducer restores order to SIGNED, rule 5 de-dups the escalation by key and returns -> no task, no event,
    no open escalation, case ACTIVE -> STUCK (liveness violation) instead of an explicit blocked/terminal state."""
    s = consented_sim()
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    s.add(ET.VISIT_NOTE_RECEIVED, note("n1"))
    s.add(ET.LCD_EVALUATED, {"checklist": ALL_MET, "missing": []})
    s.add(ET.SWO_SIGNED_RECEIVED, {"swo": SWO_OK.model_copy(update={"hcpcs": "K0003"})})
    s.apply_planned_events()
    eid = next(iter(s.state.escalations))
    for e in resolve(s.state, eid, "proceed_new_code_manual", None, "adv", s.now):
        s._apply(e)
    dec = s.plan()
    order_tasks = [t for t in dec.tasks if t.party_id == "pcp"]
    open_esc = [x for x in s.state.escalations.values() if x.status == "OPEN"]
    # F1: a terminal case (CASE_CANCELLED "handled manually by advocate") is an explicit, clean end -> not STUCK
    terminal = s.state.case.state in ("CANCELLED", "DONE")
    blocked = s.state.order.state == OrderStatus.BLOCKED_ON_HUMAN
    assert terminal or order_tasks or dec.events or open_esc or blocked, (
        f"order {s.state.order.state}: nothing planned and nothing open"
    )


def test_v1_pcp_silence_never_escalates_without_commitment_broken():
    """INTEGRATION HAZARD: ORDER_STALLED needs Commitment.followups to reach 3, but the reducer bumps it only on
    COMMITMENT_BROKEN. If a follow-up call is unanswered/fails (no BROKEN emitted), the planner re-desires the
    same followup_order_1 key forever (already DONE/FAILED in the outbox) and never escalates."""
    s = consented_sim()
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    due = s.now + 24 * H
    s.add(ET.COMMITMENT_MADE, {"cid": "pcp-1", "party": "pcp", "what": "fax", "due_at": due})
    k1 = {k for k in keys(s.at(add_business_hours(due, 24)).plan()) if k[0] == "pcp"}
    dec = s.at(add_business_hours(due, 200)).plan()
    k2 = {k for k in keys(dec) if k[0] == "pcp"}
    trig = [e.parsed().escalation.trigger for e in dec.events if e.type == ET.ESCALATION_OPENED]
    assert k2 != k1 or trig, f"200 business hours of silence: still only {k2}, no ORDER_STALLED"


def test_v1_resend_with_reused_cid_gets_no_followup():
    """INTEGRATION HAZARD: never-got-it resend. If interpret reuses the cid for the v2 promise, follow-up
    version = index(cid) = 1 and k = followups+1 = 1 -> idem key identical to the DONE v1 follow-up."""
    s = consented_sim()
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    due = s.now + 24 * H
    s.add(ET.COMMITMENT_MADE, {"cid": "pcp-order", "party": "pcp", "what": "fax", "due_at": due})
    # first follow-up rung (8 business hours after the promise)
    first = {k for k in keys(s.at(add_business_hours(due, 8)).plan()) if k[1].startswith("followup")}
    s.add(ET.ORDER_REQUEST_VERIFIED, {"received": False})
    assert ("pcp", "request_order", 2) in keys(s.plan())
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q2", "version": 2})
    due2 = s.now + 24 * H
    s.add(ET.COMMITMENT_MADE, {"cid": "pcp-order", "party": "pcp", "what": "fax", "due_at": due2})
    second = {k for k in keys(s.at(add_business_hours(due2, 8)).plan()) if k[1].startswith("followup")}
    assert second - first, f"v2 follow-up key {second} collides with v1 {first}"


def test_v1_never_got_it_with_fresh_cid_works():
    s = consented_sim()
    s.add(ET.ORDER_REQUESTED, {"request_ref": "q1", "version": 1})
    due = s.now + 24 * H
    s.add(ET.COMMITMENT_MADE, {"cid": "c1", "party": "pcp", "what": "fax", "due_at": due})
    s.at(add_business_hours(due, 8))  # first follow-up rung: next business morning
    assert ("pcp", "followup_order_1", 1) in keys(s.plan())
    s.add(ET.ORDER_REQUEST_VERIFIED, {"received": False})
    assert ("pcp", "request_order", 2) in keys(s.plan())


def test_v1_ghost_release_at_lease_end_redirects_and_recommits():
    s, a, b = happy_until_received()
    s.at(s.now + 80 * H)
    dec = s.apply_planned_events()
    assert [e.type for e in dec.events][:2] == [ET.SUPPLIER_RELEASED, ET.ORDER_REDIRECT]
    assert s.state.order.state == OrderStatus.VALID
    assert (b, "commit", 1) in keys(s.plan())


def test_v1_plan_idempotent():
    s, _, _ = happy_until_received()
    assert s.plan() == s.plan()


def test_v1_cost_concern_after_disclosure_resolves():
    from miramace.escalation import open_escalation
    from miramace.models import EscalationTrigger

    s = consented_sim()
    s.add(ET.COST_DISCLOSED, cost())
    s.add(ET.CONCERN_RAISED, {"kind": "cost", "utterance": "too much"})
    s._apply(open_escalation(EscalationTrigger.COST_CONCERN, s.state, s.now, {"utterance": "x"}))
    eid = next(iter(s.state.escalations))
    for e in resolve(s.state, eid, "nurse_resolved_accepted", None, "nurse", s.now):
        s._apply(e)
    assert s.state.patient.state == PatientStatus.COST_ACCEPTED
