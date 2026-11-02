"""Verifier helper: drive reducer + planner by hand (emulating the not-yet-written executor)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from miramace.clock import SIM_START
from miramace.events import Event, EventType, Source, make_event
from miramace.models import (
    SWO,
    Address,
    CostInfo,
    LCDChecklist,
    LCDCriterion,
    SupplierFacts,
    VisitNote,
    dr_chen,
    eleanor,
)
from miramace.planner import plan
from miramace.reducer import initial_state, reduce

ET = EventType
CASE = "v1"
ADDR = Address.parse("2100 N Lincoln Park W, Chicago, IL 60614")
GOOD = SupplierFacts(
    taking_new_medicare=True,
    k0001_in_stock=True,
    delivers_to_zip=True,
    accepts_assignment_all_months=True,
    earliest_delivery_days=2,
)
ALL_MET = LCDChecklist(results={c: "met" for c in LCDCriterion}, evidence={c: "q" for c in LCDCriterion})
SWO_OK = SWO(
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


class Sim:
    def __init__(self) -> None:
        self.now = SIM_START
        self.events: list[Event] = []
        e = make_event(CASE, ET.CASE_OPENED, {"patient": eleanor(), "pcp": dr_chen()}, self._src(), self.now)
        self.state = initial_state(CASE, eleanor(), dr_chen())
        self._apply(e)

    @staticmethod
    def _src(kind: str = "SYSTEM") -> Source:
        return Source(kind=kind, ref="verifier")

    def _apply(self, e: Event) -> None:
        e = e.model_copy(update={"seq": len(self.events) + 1})
        self.state = reduce(self.state, e)
        self.events.append(e)

    def at(self, t: datetime) -> Sim:
        self.now = t
        return self

    def add(self, t: ET, payload=None) -> Sim:
        self._apply(make_event(CASE, t, payload, self._src(), self.now))
        return self

    def plan(self):
        return plan(self.state, self.now)

    def apply_planned_events(self):
        dec = self.plan()
        for e in dec.events:
            self._apply(e)
        return dec


def keys(dec) -> set[tuple[str, str, int]]:
    return {(t.party_id, t.intent, t.version) for t in dec.tasks}


def intents(dec) -> set[str]:
    return {t.intent for t in dec.tasks}


def consented_sim() -> Sim:
    s = Sim()
    s.add(ET.PATIENT_NOTIFIED, {"purpose": "welcome_intake", "text": "hi", "grounding_ok": True, "answered": True})
    s.add(ET.ADDRESS_CONFIRMED, {"address": ADDR})
    s.add(ET.CONSENT_GIVEN, {"scope": "x"})
    s.apply_planned_events()  # CANDIDATES_RANKED
    return s


def note(doc_id: str, text: str = "note") -> dict:
    return {"note": VisitNote(doc_id=doc_id, text=text, received_at=SIM_START)}


def cost() -> dict:
    return {"cost": CostInfo()}


H = timedelta(hours=1)
