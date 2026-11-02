"""WP0 — virtual clock and business-hours / patient-hours arithmetic (America/Chicago).

Business hours: Mon–Fri, 09:00 <= t < 17:00 CT, no holiday calendar.
add_business_hours never returns a time outside business hours: a result landing exactly on 17:00 rolls to
the next business day's 09:00 (same business-time value).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from miramace.models import CHICAGO, PatientProfile

SIM_START = datetime(2026, 11, 2, 10, 0, tzinfo=CHICAGO)
BUSINESS_OPEN = 9
BUSINESS_CLOSE = 17
DEFAULT_PATIENT_HOURS = (9, 20)


class ClockError(ValueError):
    """Raised when the virtual clock would move backwards."""


def _ct(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("datetime must be tz-aware")
    return t.astimezone(CHICAGO)


def _at(day: datetime, hour: int) -> datetime:
    return datetime.combine(day.date(), time(hour), tzinfo=CHICAGO)


def is_business(t: datetime) -> bool:
    t = _ct(t)
    return t.weekday() < 5 and BUSINESS_OPEN <= t.hour < BUSINESS_CLOSE


def next_business_time(t: datetime) -> datetime:
    """t itself if within business hours, else the next business-day 09:00 CT."""
    t = _ct(t)
    if is_business(t):
        return t
    day = t if (t.weekday() < 5 and t.hour < BUSINESS_OPEN) else t + timedelta(days=1)
    day = _at(day, BUSINESS_OPEN)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return _at(day, BUSINESS_OPEN)


def add_business_hours(t: datetime, h: float) -> datetime:
    """Business-hours arithmetic; spans nights and weekends. Starts from next_business_time(t)."""
    if h < 0:
        raise ValueError("h must be >= 0")
    cur = next_business_time(t)
    remaining = timedelta(hours=h)
    while True:
        close = _at(cur, BUSINESS_CLOSE)
        avail = close - cur
        if remaining < avail:
            return cur + remaining
        remaining -= avail
        cur = next_business_time(close)
        if remaining == timedelta(0):
            return cur


def next_patient_time(t: datetime, profile: PatientProfile | None = None) -> datetime:
    """t if inside the patient's best_hours (any day), else the next start of that window."""
    start, end = profile.best_hours if profile else DEFAULT_PATIENT_HOURS
    t = _ct(t)
    if start <= t.hour < end:
        return t
    if t.hour < start:
        return _at(t, start)
    return _at(t + timedelta(days=1), start)


def is_patient_time(t: datetime, profile: PatientProfile | None = None) -> bool:
    return next_patient_time(t, profile) == _ct(t)


class VirtualClock:
    """Monotonic simulated clock. Never moves backwards."""

    def __init__(self, start: datetime = SIM_START) -> None:
        self._now = _ct(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, t: datetime) -> datetime:
        t = _ct(t)
        if t < self._now:
            raise ClockError(f"clock cannot move backwards: {t.isoformat()} < {self._now.isoformat()}")
        self._now = t
        return self._now

    def advance(self, seconds: float) -> datetime:
        return self.advance_to(self._now + timedelta(seconds=seconds))
