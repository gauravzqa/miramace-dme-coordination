from datetime import datetime

import pytest

from miramace.clock import (
    SIM_START,
    ClockError,
    VirtualClock,
    add_business_hours,
    is_business,
    next_business_time,
    next_patient_time,
)
from miramace.models import CHICAGO, PatientProfile


def ct(*a):
    return datetime(*a, tzinfo=CHICAGO)


def test_sim_start():
    assert VirtualClock().now() == ct(2026, 11, 2, 10, 0) == SIM_START
    assert SIM_START.weekday() == 0


def test_fri_plus_two_business_hours_is_monday():
    assert add_business_hours(ct(2026, 11, 6, 16), 2) == ct(2026, 11, 9, 10)


def test_add_business_hours_more():
    assert add_business_hours(ct(2026, 11, 2, 10), 3) == ct(2026, 11, 2, 13)
    assert add_business_hours(ct(2026, 11, 2, 10), 24) == ct(2026, 11, 5, 10)
    assert add_business_hours(ct(2026, 11, 2, 15), 2) == ct(2026, 11, 3, 9)  # lands on close -> next open
    assert add_business_hours(ct(2026, 11, 7, 12), 1) == ct(2026, 11, 9, 10)  # Saturday start


def test_is_business_and_next_business_time():
    assert is_business(ct(2026, 11, 2, 9))
    assert not is_business(ct(2026, 11, 2, 17))
    assert not is_business(ct(2026, 11, 7, 12))
    assert next_business_time(ct(2026, 11, 7, 12)) == ct(2026, 11, 9, 9)
    assert next_business_time(ct(2026, 11, 2, 7)) == ct(2026, 11, 2, 9)
    assert next_business_time(ct(2026, 11, 6, 18)) == ct(2026, 11, 9, 9)


def test_next_patient_time():
    assert next_patient_time(ct(2026, 11, 2, 21)) == ct(2026, 11, 3, 9)
    assert next_patient_time(ct(2026, 11, 2, 8)) == ct(2026, 11, 2, 9)
    assert next_patient_time(ct(2026, 11, 7, 12)) == ct(2026, 11, 7, 12)
    assert next_patient_time(ct(2026, 11, 2, 18), PatientProfile(best_hours=(10, 18))) == ct(2026, 11, 3, 10)


def test_advance_to_is_monotonic():
    c = VirtualClock()
    c.advance_to(ct(2026, 11, 2, 11))
    c.advance_to(ct(2026, 11, 2, 11))
    with pytest.raises(ClockError):
        c.advance_to(ct(2026, 11, 2, 10))
    assert c.now() == ct(2026, 11, 2, 11)
