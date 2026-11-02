from datetime import datetime

import pytest
from pydantic import BaseModel, ValidationError

from miramace.clock import SIM_START
from miramace.events import (
    PAYLOADS,
    EventType,
    OrderRequested,
    PlannerDecision,
    Source,
    SupplierQualified,
    make_event,
)
from miramace.models import SupplierFacts

SRC = Source(kind="RULE", ref="test")


def test_every_event_type_has_payload():
    assert set(PAYLOADS) == set(EventType)
    for et, model in PAYLOADS.items():
        assert issubclass(model, BaseModel), et


def test_make_event_from_model_and_dict():
    req = OrderRequested(request_ref="r1", version=1)
    e = make_event("c1", EventType.ORDER_REQUESTED, req, SRC, SIM_START)
    assert e.seq == 0 and e.payload == {"request_ref": "r1", "version": 1}
    payload = {"sid": "s", "facts": SupplierFacts().model_dump(), "verified_at": SIM_START}
    e2 = make_event("c1", "SUPPLIER_QUALIFIED", payload, SRC, SIM_START)
    parsed = e2.parsed()
    assert isinstance(parsed, SupplierQualified) and parsed.verified_at == SIM_START
    assert make_event("c1", EventType.CASE_DONE, None, SRC, SIM_START).payload == {}


def test_make_event_rejects_bad_payload():
    with pytest.raises(ValidationError):
        make_event("c1", EventType.ORDER_REQUESTED, {"request_ref": "r1"}, SRC, SIM_START)  # missing version
    with pytest.raises(ValidationError):
        make_event("c1", EventType.CASE_DONE, {"unexpected": 1}, SRC, SIM_START)
    with pytest.raises(TypeError):
        make_event("c1", EventType.ORDER_SENT, OrderRequested(request_ref="r", version=1), SRC, SIM_START)
    with pytest.raises(ValueError):
        make_event("c1", "NOT_AN_EVENT", {}, SRC, SIM_START)


def test_sim_time_must_be_tz_aware():
    with pytest.raises(ValidationError):
        make_event("c1", EventType.CASE_DONE, None, SRC, datetime(2026, 11, 2, 10))


def test_planner_decision_roundtrip():
    e = make_event("c1", EventType.CASE_DONE, None, SRC, SIM_START)
    d = PlannerDecision(events=[e])
    assert PlannerDecision.model_validate_json(d.model_dump_json()).events[0].type == EventType.CASE_DONE
