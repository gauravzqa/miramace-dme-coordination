from datetime import date

import pytest
from pydantic import ValidationError

from miramace.models import (
    Address,
    CaseState,
    Claim,
    LCDChecklist,
    LCDCriterion,
    PersonaSpec,
    dr_chen,
    eleanor,
    idem_key,
    load_suppliers,
)


def test_eleanor():
    p = eleanor()
    assert p.name == "Eleanor Martinez" and p.age == 72
    assert p.dob == date(1954, 3, 14)
    assert p.address is None and p.medigap is False and p.coverage == "ORIGINAL_PART_B"
    assert p.phi().mbi == "1EG4-TE5-MK72"
    assert p.profile.best_hours == (9, 20)


def test_dr_chen():
    d = dr_chen()
    assert d.name == "Dr. Sarah Chen" and d.practice == "Sunrise Family Medicine"
    assert d.phone == "(312) 555-0198" and d.party_id == "pcp"


def test_load_suppliers():
    sups = load_suppliers()
    assert len(sups) == 12
    sids = [s.sid for s in sups]
    assert len(set(sids)) == 12
    assert sids[0] == "lakeshore-home-medical-equipment"
    assert "windy-city-mobility-dme" in sids
    by = {s.sid: s for s in sups}
    assert by["lakeshore-home-medical-equipment"].zip == "60614"
    assert by["midwest-mobility-solutions"].address.city == "Niles"
    assert by["evanston-home-medical"].zip == "60201"
    assert all(len(s.zip) == 5 and s.zip.isdigit() for s in sups)


def test_address_parse():
    a = Address.parse("1820 N Clark St, Chicago, IL 60614")
    assert (a.line1, a.city, a.state, a.zip) == ("1820 N Clark St", "Chicago", "IL", "60614")
    with pytest.raises(ValueError):
        Address.parse("nowhere")


def test_idem_key_format():
    assert idem_key("c1", "pcp", "request_order", 2) == "c1:pcp:request_order:v2"


def test_case_state_defaults_and_lcd():
    s = CaseState(case_id="c1")
    assert s.case.state == "ACTIVE" and s.patient.state == "INTAKE" and s.order.state == "NOT_SENT"
    assert s.track("delivery").state == "NOT_STARTED"
    lcd = LCDChecklist(results={c: "met" for c in LCDCriterion})
    assert lcd.all_met
    lcd.results[LCDCriterion.HOME_ACCOMMODATES] = "not_documented"
    assert lcd.missing == [LCDCriterion.HOME_ACCOMMODATES]


def test_persona_spec_roundtrip():
    spec = PersonaSpec.model_validate(
        {
            "party_id": "pcp",
            "role": "pcp",
            "org_name": "Sunrise Family Medicine",
            "answerer_name": "Maria",
            "facts": [{"id": "f1", "slot": "fax_number", "value": "x", "text": "x", "reveal_when": "asked"}],
            "sim": {"ghost_after": "commit", "note_claims": [Claim(field="a", status="unknown", value=None,
                                                                      evidence_quote=None).model_dump()]},
        }
    )
    assert spec.behaviour == "helpful" and spec.sim.note_delay_bh == 16
    with pytest.raises(ValidationError):
        PersonaSpec(party_id="x", role="pcp", org_name="o", answerer_name="a", behaviour="rude")
