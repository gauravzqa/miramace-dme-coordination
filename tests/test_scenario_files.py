"""B7 — scenario YAML files: schema sanity (always) + end-to-end expectations (once the B6 engine exists)."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
import yaml

from miramace.models import Address, EscalationTrigger, PersonaSpec, load_suppliers

SCEN_DIR = Path(__file__).resolve().parents[1] / "scenarios"
B7_SCENARIOS = [
    "supplier_disqualified",
    "ghost_failover",
    "pcp_never_got_it",
    "lcd_missing_addendum",
    "cost_concern",
    "code_drift_k0003",
]
ALL_FILES = sorted(p.stem for p in SCEN_DIR.glob("*.yaml"))
SIDS = {s.sid for s in load_suppliers()}
PURPOSES = {"welcome_intake", "cost_disclosure", "delivery_slot", "delivered_checkin", "status_update"}


def _load(name: str) -> dict:
    return yaml.safe_load((SCEN_DIR / f"{name}.yaml").read_text())


def _engine_ready() -> bool:
    try:
        mod = importlib.import_module("miramace.scenario")
    except Exception:
        return False
    return all(hasattr(mod, a) for a in ("run_scenario", "load_scenario"))


@pytest.mark.parametrize("name", B7_SCENARIOS)
def test_b7_scenario_file_exists(name: str) -> None:
    assert (SCEN_DIR / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", ALL_FILES)
def test_scenario_schema(name: str) -> None:
    d = _load(name)
    assert d["name"] == name
    assert d.get("description")
    Address.parse(d["patient"]["address"])
    for purpose, intents in d["patient_script"].items():
        assert purpose in PURPOSES, purpose
        assert intents, purpose
    personas = d["personas"]
    assert "pcp" in personas
    for party_id, spec in personas.items():
        p = PersonaSpec.model_validate({"party_id": party_id, **spec})
        if p.role == "supplier":
            assert party_id in SIDS, f"unknown supplier sid {party_id}"
        else:
            assert party_id == "pcp"
        ids = [f.id for f in p.facts]
        assert len(ids) == len(set(ids)), f"duplicate fact ids in {party_id}"
        sim = p.sim
        if sim.note_claims:
            assert sim.visit_note_text, f"{party_id}: note_claims need visit_note_text"
            for c in sim.note_claims:
                assert c.evidence_quote and c.evidence_quote in sim.visit_note_text, c.field
        if sim.addendum_claims:
            assert sim.addendum_text
            for c in sim.addendum_claims:
                assert c.evidence_quote and c.evidence_quote in sim.addendum_text, c.field
    exp = d["expect"]
    assert exp["final"] in ("DONE", "BLOCKED_ON_HUMAN")
    for t in exp.get("escalations", []):
        EscalationTrigger(t)
    assert exp.get("max_sim_days", 1) > 0
    gt = d["ground_truth"]
    assert {"claim_would_pass", "lcd_supported", "hcpcs"} <= set(gt)


def test_lcd_missing_addendum_really_misses_home() -> None:
    sim = _load("lcd_missing_addendum")["personas"]["pcp"]["sim"]
    assert "HOME_ACCOMMODATES" not in {c["field"] for c in sim["note_claims"]}
    assert "HOME_ACCOMMODATES" in {c["field"] for c in sim["addendum_claims"]}


@pytest.mark.skipif(not _engine_ready(), reason="B6 scenario engine not available yet")
@pytest.mark.parametrize("name", ALL_FILES)
def test_scenario_meets_expectation(name: str, tmp_path: Path) -> None:
    from miramace.scenario import run_scenario

    summary, store, scen = asyncio.run(run_scenario(name, mode="fake", db_dir=str(tmp_path)))
    exp = _load(name)["expect"]
    assert str(summary.status) == exp["final"], (summary.status, summary.escalations)
    assert sorted(summary.escalations) == sorted(exp.get("escalations", [])), summary.escalations
    assert not summary.invariant_violations
    days = (summary.sim_end - summary.sim_start).total_seconds() / 86400
    assert days <= exp.get("max_sim_days", 14)
