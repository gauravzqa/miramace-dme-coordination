from datetime import date, timedelta

import pytest

from miramace.clock import SIM_START
from miramace.models import (
    SWO,
    AssignmentStatus,
    CaseState,
    Claim,
    LCDChecklist,
    LCDCriterion,
    OrderStatus,
    PatientStatus,
    SupplierFacts,
    SupplierRecord,
    SupplierStatus,
    TrackState,
    load_suppliers,
)
from miramace.rules import (
    can_schedule,
    check_code,
    cost_info,
    facts_consistent,
    lcd_evaluate,
    lcd_missing_wording,
    parse_bool,
    rank_suppliers,
    supplier_eligible,
    swo_completeness,
)


def full_swo(**kw) -> SWO:
    base = dict(
        beneficiary_name="Eleanor Martinez",
        mbi="1EG4-TE5-MK72",
        order_date=date(2026, 11, 3),
        item_description="Standard manual wheelchair",
        hcpcs="K0001",
        quantity=1,
        practitioner_name="Dr. Sarah Chen",
        practitioner_npi="1234567893",
        signed=True,
        signature_date=date(2026, 11, 3),
    )
    base.update(kw)
    return SWO(**base)


def test_rank_from_60614():
    ranked = rank_suppliers(load_suppliers(), "60614")
    assert [sid for sid, _ in ranked[:2]] == [
        "lakeshore-home-medical-equipment",
        "lincoln-park-dme-services",
    ]
    miles = [m for _, m in ranked]
    assert miles == sorted(miles)
    assert len(ranked) == 12


def test_check_code():
    assert check_code("K0001") == "OK"
    assert check_code(" k0001 ") == "OK"
    assert check_code("K0003") == "DRIFT"
    assert check_code("E1130") == "DRIFT"
    assert check_code(None) == "MISSING"
    assert check_code("") == "MISSING"


def test_swo_missing_npi():
    assert swo_completeness(full_swo()) == []
    assert swo_completeness(full_swo(practitioner_npi=None)) == ["practitioner_npi"]
    assert set(swo_completeness(full_swo(signed=False, signature_date=None))) == {"signed", "signature_date"}
    assert "beneficiary_name" in swo_completeness(None)


def test_supplier_eligible():
    good = SupplierFacts(
        taking_new_medicare=True,
        k0001_in_stock=True,
        delivers_to_zip=True,
        accepts_assignment_all_months=True,
    )
    assert supplier_eligible(good) == (True, None)
    assert supplier_eligible(good.model_copy(update={"delivers_to_zip": None})) == (
        False,
        "unknown:delivers_to_zip",
    )
    # definite False beats unknown
    f = good.model_copy(update={"taking_new_medicare": None, "k0001_in_stock": False})
    assert supplier_eligible(f) == (False, "not:k0001_in_stock")
    assert supplier_eligible(SupplierFacts())[0] is False


def test_facts_consistent():
    assert facts_consistent(SupplierFacts(k0001_in_stock=True, earliest_delivery_days=2)) == []
    assert facts_consistent(SupplierFacts(k0001_in_stock=False, earliest_delivery_days=2))


def test_lcd_evaluate():
    claims = [
        Claim(field="MRADL_LIMITATION", status="stated", value="yes", evidence_quote="cannot reach bathroom"),
        Claim(field="cane_walker_insufficient", status="stated", value="met", evidence_quote="walker unsafe"),
        Claim(field="HOME_ACCOMMODATES", status="stated", value="true", evidence_quote=None),  # no quote
        Claim(field="WILL_USE_IMPROVES", status="unknown", value="yes", evidence_quote="x"),
        Claim(field="SELF_PROPEL_OR_CAREGIVER", status="stated", value="no", evidence_quote="cannot propel"),
    ]
    cl = lcd_evaluate(claims)
    assert cl.results[LCDCriterion.MRADL_LIMITATION] == "met"
    assert cl.results[LCDCriterion.CANE_WALKER_INSUFFICIENT] == "met"
    assert cl.results[LCDCriterion.HOME_ACCOMMODATES] == "not_documented"
    assert cl.results[LCDCriterion.WILL_USE_IMPROVES] == "not_documented"
    assert cl.results[LCDCriterion.SELF_PROPEL_OR_CAREGIVER] == "not_met"
    assert cl.evidence[LCDCriterion.MRADL_LIMITATION] == "cannot reach bathroom"
    assert not cl.all_met
    assert set(cl.missing) == {
        LCDCriterion.HOME_ACCOMMODATES,
        LCDCriterion.WILL_USE_IMPROVES,
        LCDCriterion.SELF_PROPEL_OR_CAREGIVER,
    }


def test_lcd_wording_all():
    for c in LCDCriterion:
        assert len(lcd_missing_wording(c)) > 20
    with pytest.raises(ValueError):
        lcd_missing_wording("nope")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True),
        ("Yep, we do.", True),
        ("we do", True),
        ("Absolutely", True),
        ("no", False),
        ("not at the moment", False),
        ("Unfortunately not", False),
        ("We don't, sorry", False),
        ("no problem", True),
        ("hmm, let me check", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_bool(text, expected):
    assert parse_bool(text) is expected


def test_cost_info_constants():
    c = cost_info()
    assert (c.deductible_usd, c.coinsurance_pct, c.rental_months) == (283, 20, 13)


def ready_state(**kw) -> CaseState:
    sid = "lakeshore-home-medical-equipment"
    all_met = LCDChecklist(results={c: "met" for c in LCDCriterion})
    base = dict(
        case_id="c1",
        patient=TrackState(state=PatientStatus.COST_ACCEPTED),
        order=TrackState(state=OrderStatus.RECEIVED_BY),
        committed_sid=sid,
        order_received_by=sid,
        swo=full_swo(),
        lcd=all_met,
        assignment_by_sid={sid: AssignmentStatus.CONFIRMED_ALL_MONTHS},
        suppliers={sid: SupplierRecord(sid=sid, status=SupplierStatus.COMMITTED, verified_at=SIM_START)},
    )
    base.update(kw)
    return CaseState(**base)


def test_can_schedule():
    ok, reasons = can_schedule(ready_state(), SIM_START + timedelta(hours=2))
    assert ok and reasons == []
    ok, reasons = can_schedule(ready_state(), SIM_START + timedelta(hours=25))
    assert not ok and any("24h" in r for r in reasons)
    ok, reasons = can_schedule(ready_state(swo=full_swo(hcpcs="K0003")), SIM_START)
    assert not ok
    ok, reasons = can_schedule(ready_state(order_received_by="other"), SIM_START)
    assert not ok
    ok, reasons = can_schedule(ready_state(patient=TrackState(state=PatientStatus.COST_DISCLOSED)), SIM_START)
    assert not ok
    ok, reasons = can_schedule(ready_state(assignment_by_sid={}), SIM_START)
    assert not ok


@pytest.mark.parametrize(
    "text", ["not sure", "I don't know", "maybe", "let me check", "Maybe, I'd have to check.", "No idea"]
)
def test_parse_bool_hedges_are_unknown(text):
    assert parse_bool(text) is None
