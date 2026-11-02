"""B6 Engine — scenario YAML loader and harness.

YAML schema (validated by `Scenario`; full guide in scenarios/README.md):

    name: happy_path                       # required; also the file stem
    description: one line
    patient:                               # PatientSpec
      address: "2150 N Lincoln Park W, Chicago, IL 60614"   # what the patient will confirm (info only)
      best_hours: [9, 20]
      caregiver_name: Ana                  # optional: caregiver_phone, home_access, hearing_or_language,
                                           #           prior_concerns: [..]
    patient_script:                        # fakes.patient.ScriptedPatient format, keyed by NotifyPurpose
      welcome_intake: [{kind: confirm_address, address: "..."}, {kind: give_consent}]
      cost_disclosure: [{kind: cost_accepted}]
      delivery_slot: [{kind: confirm_delivery_slot}]
      delivered_checkin: [{kind: delivery_confirmed}]
    personas:                              # party_id -> models.PersonaSpec (party_id filled from the key)
      pcp: {role: pcp, org_name: Sunrise Family Medicine, answerer_name: Maria, behaviour: helpful,
            facts: [{id, slot, value, text, reveal_when}], sim: {...models.PersonaSim...}}
      <supplier sid>: {role: supplier, org_name: <exact directory name>, ...}
                                           # suppliers not listed never answer (NO_ANSWER)
    ground_truth: {claim_would_pass: true, lcd_supported: true, hcpcs: K0001}
    expect: {final: DONE, escalations: [], max_sim_days: 14}

Claims inside `sim.note_claims` / `sim.addendum_claims` may omit `status` (default "stated"), `value`
(default "met") and `evidence_quote` (default null). Supplier `org_name` must equal the directory name
(data/suppliers.csv) and the PCP's must equal "Sunrise Family Medicine" — the caller verifies the party by it.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from miramace.clock import SIM_START, VirtualClock
from miramace.events import Event, EventType, Source, make_event
from miramace.models import (
    PCP_PARTY_ID,
    PatientProfile,
    PersonaSpec,
    dr_chen,
    eleanor,
    load_suppliers,
)
from miramace.ports import Ports
from miramace.store import Store

log = logging.getLogger(__name__)

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"

Final = Literal["DONE", "BLOCKED_ON_HUMAN", "STUCK", "MAX_STEPS", "CANCELLED"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientSpec(_M):
    address: str | None = None
    best_hours: tuple[int, int] = (9, 20)
    caregiver_name: str | None = None
    caregiver_phone: str | None = None
    home_access: str | None = None
    hearing_or_language: str | None = None
    prior_concerns: list[str] = Field(default_factory=list)

    def profile(self) -> PatientProfile:
        return PatientProfile(
            best_hours=self.best_hours,
            caregiver_name=self.caregiver_name,
            caregiver_phone=self.caregiver_phone,
            hearing_or_language=self.hearing_or_language,
            prior_concerns=list(self.prior_concerns),
            home_access=self.home_access,
        )


class GroundTruth(_M):
    claim_would_pass: bool
    lcd_supported: bool = True
    hcpcs: str = "K0001"


class Expect(_M):
    final: Final = "DONE"
    escalations: list[str] = Field(default_factory=list)  # EscalationTrigger values
    max_sim_days: float | None = None


def _fill_claims(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for c in raw or []:
        c = dict(c)
        c.setdefault("status", "stated")
        c.setdefault("value", "met")
        c.setdefault("evidence_quote", None)
        if c["value"] is not None:
            c["value"] = str(c["value"])
        out.append(c)
    return out


class Scenario(_M):
    name: str
    description: str = ""
    patient: PatientSpec = Field(default_factory=PatientSpec)
    patient_script: dict[str, Any] = Field(default_factory=dict)
    personas: dict[str, PersonaSpec] = Field(default_factory=dict)
    ground_truth: GroundTruth
    expect: Expect = Field(default_factory=Expect)
    source: str | None = None  # file path it was loaded from (set by load_scenario)

    @model_validator(mode="before")
    @classmethod
    def _prep(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        personas = {}
        for pid, spec in (data.get("personas") or {}).items():
            if isinstance(spec, dict):
                spec = dict(spec)
                spec.setdefault("party_id", pid)
                sim = dict(spec.get("sim") or {})
                for k in ("note_claims", "addendum_claims"):
                    if k in sim:
                        sim[k] = _fill_claims(sim[k])
                for k, v in list(sim.items()):  # yaml may give numbers for text-ish fields
                    if k == "ghost_after" and v is False:
                        sim[k] = "never"
                spec["sim"] = sim
                for f in spec.get("facts") or []:
                    if isinstance(f, dict) and f.get("value") is not None:
                        f["value"] = str(f["value"])
            personas[pid] = spec
        data["personas"] = personas
        return data

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        directory = {s.sid: s.name for s in load_suppliers()}
        for pid, spec in self.personas.items():
            if spec.party_id != pid:
                raise ValueError(f"persona key {pid!r} != party_id {spec.party_id!r}")
            if pid == PCP_PARTY_ID:
                if spec.role != "pcp":
                    raise ValueError("persona 'pcp' must have role pcp")
                if spec.org_name != dr_chen().practice:
                    raise ValueError(f"pcp org_name must be {dr_chen().practice!r}")
            else:
                if spec.role != "supplier":
                    raise ValueError(f"persona {pid!r} must have role supplier")
                if pid not in directory:
                    raise ValueError(f"persona {pid!r} is not a directory supplier sid")
                if spec.org_name != directory[pid]:
                    raise ValueError(f"persona {pid!r} org_name must be {directory[pid]!r}")
        return self


# ------------------------------------------------------------------------------------------ loading


def _resolve(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = SCENARIO_DIR / f"{name_or_path}.yaml"
    if cand.exists():
        return cand
    if p.exists():
        return p
    raise FileNotFoundError(f"scenario {name_or_path!r} not found (looked in {SCENARIO_DIR})")


def load_scenario(name_or_path: str | Path) -> Scenario:
    path = _resolve(name_or_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sc = Scenario.model_validate(data)
    return sc.model_copy(update={"source": str(path)})


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


# ------------------------------------------------------------------------------------------ ports


def build_ports(
    scenario: Scenario,
    store: Store,
    clock: VirtualClock,
    *,
    mode: Literal["fake", "live"] = "fake",
    voice_pcp: bool = False,
) -> Ports:
    """Wire the simulated world (personas, faxes, scripted patient) to fake or live AI components."""
    del store
    from miramace.calls.brains import FakeCallerBrain, FakePersonaBrain, LLMCallerBrain, LLMPersonaBrain
    from miramace.calls.runner import TextCallerPort
    from miramace.extract.extractor import Extractor, FakeExtractor
    from miramace.fakes.companion import KeywordCompanion
    from miramace.fakes.patient import ScriptedPatient
    from miramace.fakes.sim import (
        EffectivePersonas,
        FactSheetDocReader,
        InboundSimulator,
        LLMDocReader,
        SimCallerPort,
    )

    personas = dict(scenario.personas)
    llm = None
    if mode == "live":
        from miramace.companion.agent import LLMCompanion
        from miramace.llm.port import make_llm

        llm = make_llm("live")  # OpenAILLM; raises LLMUnavailable with a clear message if env is missing

    def brains(eff: EffectivePersonas) -> tuple[Any, Any, Any]:
        """(caller factory, persona factory(pid), extractor) — live LLM brains or offline rule-based ones."""
        if llm is None:
            return (lambda: FakeCallerBrain()), (lambda pid: FakePersonaBrain(eff[pid])), FakeExtractor(eff)
        return (lambda: LLMCallerBrain(llm)), (lambda pid: LLMPersonaBrain(llm, eff[pid])), Extractor(llm)

    def make_caller() -> SimCallerPort:
        sim = SimCallerPort(personas, None, clock)
        caller_f, persona_f, extractor = brains(EffectivePersonas(sim))
        sim.inner = TextCallerPort(caller_f, persona_f, extractor, clock=clock)
        return sim

    caller_pcp = make_caller()
    caller_supplier = make_caller()
    if voice_pcp:
        # --voice-pcp is explicit: enable the pcp voice leg whatever .env says. VoiceCallerPort itself falls
        # back to text (transport="text_fallback", flag voice_failed:<reason>) on ImportError/LiveKit errors.
        os.environ["MIRA_VOICE"] = "1"
        legs = {x.strip() for x in os.environ.get("MIRA_VOICE_LEGS", "pcp").split(",") if x.strip()}
        os.environ["MIRA_VOICE_LEGS"] = ",".join(sorted(legs | {"pcp"}))
        try:
            from miramace.voice.transport import VoiceCallerPort

            eff = EffectivePersonas(caller_pcp)
            caller_f, persona_f, extractor = brains(eff)
            caller_pcp.inner = VoiceCallerPort(
                caller_pcp.inner,  # the text port: fallback + non-voice legs
                caller_f,
                persona_f,
                caller_pcp.effective,  # persona lookup (YAML persona + sim defaults)
                extractor,
                clock=clock,
            )
        except Exception as exc:  # wiring error: keep text, but say so loudly
            log.warning("voice PCP leg unavailable (%s: %s); PCP calls stay on text", type(exc).__name__, exc)
            caller_pcp.voice_error = exc  # type: ignore[attr-defined]
    script = {k: v for k, v in scenario.patient_script.items()}
    return Ports(
        caller_pcp=caller_pcp,
        caller_supplier=caller_supplier,
        doc_reader=FactSheetDocReader(personas) if llm is None else LLMDocReader(llm),
        companion=KeywordCompanion()
        if llm is None
        else LLMCompanion(llm, profile=scenario.patient.profile()),
        patient=ScriptedPatient(script),
        inbound=InboundSimulator(personas, clock),
        llm=llm,
    )


# ------------------------------------------------------------------------------------------ runs


def _patient(sc: Scenario):  # -> Patient
    return eleanor().model_copy(update={"profile": sc.patient.profile()})


def seed_case(
    store: Store, scenario: Scenario, case_id: str, clock: VirtualClock, *, mode: str = "fake"
) -> Event:
    """cases row + suppliers table + CASE_OPENED (source SYSTEM)."""
    store.create_case(
        case_id,
        scenario.name,
        {"scenario": scenario.name, "source": scenario.source, "mode": mode},
        clock.now(),
    )
    store.upsert_suppliers(load_suppliers())
    ev = make_event(
        case_id,
        EventType.CASE_OPENED,
        {"patient": _patient(scenario), "pcp": dr_chen()},
        Source(kind="SYSTEM", ref="intake"),
        clock.now(),
    )
    return store.append(ev)


async def run_scenario(
    name_or_path: str | Path,
    *,
    mode: Literal["fake", "live"] = "fake",
    voice_pcp: bool = False,
    db_dir: str | Path | None = "runs",
    on_event: Any = None,
    max_steps: int = 400,
) -> tuple[Any, Store, Scenario]:
    """Load, seed and run a scenario. Returns (RunSummary, Store, Scenario). db_dir=None -> in-memory DB."""
    from miramace.runloop import run_case

    sc = load_scenario(name_or_path)
    case_id = f"{sc.name}-{uuid.uuid4().hex[:6]}"
    store = Store(":memory:" if db_dir is None else Path(db_dir) / f"{case_id}.db")
    clock = VirtualClock(SIM_START)
    ports = build_ports(sc, store, clock, mode=mode, voice_pcp=voice_pcp)
    first = seed_case(store, sc, case_id, clock, mode=mode)
    if on_event:
        on_event(first)
    summary = await run_case(case_id, ports, store, clock, on_event=on_event, max_steps=max_steps)
    return summary, store, sc


def open_case(case_id: str, db_dir: str | Path = "runs") -> tuple[Store, VirtualClock]:
    """Re-open a finished run's DB; the clock is set to the last event's sim_time."""
    path = Path(db_dir) / f"{case_id}.db"
    if not path.exists():
        raise FileNotFoundError(path)
    store = Store(path)
    events = store.events(case_id)
    clock = VirtualClock(events[-1].sim_time if events else SIM_START)
    return store, clock


def scenario_for_case(store: Store, case_id: str) -> Scenario:
    """The Scenario a stored case was run from (to rebuild its ports for chat / resolve)."""
    case = store.case(case_id)
    if case is None:
        raise KeyError(case_id)
    intake = case["intake"]
    src = intake.get("source")
    return load_scenario(src if src and Path(src).exists() else intake.get("scenario") or case["scenario"])
