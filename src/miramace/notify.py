"""B4 Patient side — notify policy (WHEN Eleanor hears something) + safe templates (fallback wording).

Code decides when; the LLM only words it (companion.compose), and the template below is the grounded fallback.
Template placeholders are filled ONLY from snapshot facts (see `template_facts`).
"""

from __future__ import annotations

from datetime import datetime

from miramace.clock import is_patient_time
from miramace.models import CHICAGO, CaseState, CaseStatus, NotifyPurpose, Snapshot, Track
from miramace.snapshot import cost_text

ACTION_REQUIRED: frozenset[str] = frozenset({"welcome_intake", "cost_disclosure", "delivery_slot"})


def should_notify(state: CaseState, purpose: str, now: datetime, policy_cap_per_day: int = 1) -> bool:
    """True if Eleanor may be contacted now for `purpose`.

    - only inside her best hours (clock.is_patient_time with her profile);
    - only while the case is ACTIVE;
    - action-required purposes (welcome_intake, cost_disclosure, delivery_slot) bypass the daily cap;
    - everything else: at most `policy_cap_per_day` per sim day. CaseState only records `last_notified_at`,
      so the effective cap is 1/day (any notification earlier the same CT day blocks an informational one).
    """
    profile = state.patient_info.profile if state.patient_info else None
    if not is_patient_time(now, profile):
        return False
    if state.case.state != CaseStatus.ACTIVE:
        return False
    if purpose in ACTION_REQUIRED:
        return True
    if policy_cap_per_day <= 0:
        return False
    last = state.last_notified_at
    if last is not None and last.astimezone(CHICAGO).date() == now.astimezone(CHICAGO).date():
        return False
    return True


TEMPLATES: dict[str, str] = {
    "welcome_intake": (
        "Hi, I'm your Mira Mace assistant. I'll take care of getting your wheelchair, and a nurse is "
        "always available if you need one. To get started, could you share your home address with the "
        "ZIP code, and let me know it's OK for us to arrange the wheelchair with {pcp_office} and an "
        "equipment company for you?"
    ),
    "cost_disclosure": (
        "Here's what to expect for cost. {cost} Does that work for you? If you have any worries about "
        "cost, just tell me and a nurse will talk it through with you."
    ),
    "delivery_slot": (
        "Good news: {delivery_status} Does that time work for you, or would you prefer another time?"
    ),
    "delivered_checkin": (
        "{delivery_status} Is it working for you, and does it fit through your doorways? Just reply to let "
        "me know."
    ),
    "status_update": "Quick update: {headline} {next_action}",
}

SAFE_REPLY = "{headline} I'll find out more and let you know. A nurse is always available if you need one."


def template_facts(snapshot: Snapshot) -> dict[str, str]:
    """Placeholder values, every one taken from the snapshot."""
    by_track = {it.track: it for it in snapshot.items}
    doc = next((n for n in snapshot.allowed_names if n.startswith("Dr. ") and len(n.split()) == 2), None)
    delivery = by_track.get(Track.DELIVERY)
    return {
        "headline": snapshot.headline,
        "next_action": snapshot.next_patient_action or "",
        "cost": cost_text(snapshot.cost),
        "pcp_office": f"{doc}'s office" if doc else "your doctor's office",
        "delivery_status": delivery.plain_status if delivery else snapshot.headline,
    }


def render_template(purpose: NotifyPurpose | str, snapshot: Snapshot) -> str:
    tpl = TEMPLATES.get(purpose, TEMPLATES["status_update"])
    return " ".join(tpl.format(**template_facts(snapshot)).split())


def safe_reply(snapshot: Snapshot) -> str:
    return " ".join(SAFE_REPLY.format(**template_facts(snapshot)).split())
