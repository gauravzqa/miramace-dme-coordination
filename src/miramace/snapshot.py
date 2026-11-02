"""B4 Patient side — case snapshot: the plain-language read model that is the companion's ONLY fact source.

`build_snapshot(state, now)` turns a CaseState into Eleanor-friendly items (one per track) with who holds the
ball and by when, plus the fixed cost explanation. The allowed_* lists enumerate every name / date rendering /
amount the snapshot contains; grounding.py checks outbound text against them.

Cost: only the constants in CostInfo ($283 deductible, 20%, 13 months) — never an invented allowed amount.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

from miramace.models import (
    CHICAGO,
    AssignmentStatus,
    CaseState,
    CaseStatus,
    CommitmentStatus,
    CostInfo,
    DeliveryStatus,
    OrderStatus,
    Owner,
    PatientStatus,
    Snapshot,
    SnapshotItem,
    SupplierTrackStatus,
    Track,
    load_suppliers,
)

SNAPSHOT_TRACKS: tuple[Track, ...] = (
    Track.ORDER,
    Track.SUPPLIER,
    Track.ASSIGNMENT,
    Track.DELIVERY,
    Track.PATIENT,
)

# --------------------------------------------------------------------------------------------- dates


def _ct(dt: datetime) -> datetime:
    return dt.astimezone(CHICAGO)


def day_label(dt: datetime) -> str:
    """'Thursday, Nov 5'."""
    d = _ct(dt)
    return f"{d.strftime('%A')}, {d.strftime('%b')} {d.day}"


def time_label(dt: datetime) -> str:
    """'10 AM' / '2:30 PM'."""
    d = _ct(dt)
    h = d.hour % 12 or 12
    ampm = "AM" if d.hour < 12 else "PM"
    return f"{h} {ampm}" if d.minute == 0 else f"{h}:{d.minute:02d} {ampm}"


def date_variants(dt: datetime, now: datetime | None = None) -> list[str]:
    """Every rendering of dt's date that the companion may use."""
    d = _ct(dt)
    wd, mon, mon_full = d.strftime("%A"), d.strftime("%b"), d.strftime("%B")
    out = [
        wd,
        f"{mon} {d.day}",
        f"{mon_full} {d.day}",
        f"{d.month}/{d.day}",
        f"{wd}, {mon} {d.day}",
        f"{wd}, {mon_full} {d.day}",
        mon_full,
    ]
    if now is not None:
        n = _ct(now).date()
        if d.date() == n:
            out.append("today")
        elif d.date() == n + timedelta(days=1):
            out.append("tomorrow")
    return out


# ---------------------------------------------------------------------------------------- labels


@lru_cache(maxsize=1)
def _directory() -> dict[str, str]:
    try:
        return {s.sid: s.name for s in load_suppliers()}
    except Exception:  # pragma: no cover - directory missing in odd environments
        return {}


def supplier_name(sid: str | None) -> str | None:
    if not sid:
        return None
    return _directory().get(sid) or sid.replace("-", " ").title()


def pcp_short(state: CaseState) -> str:
    """'Dr. Sarah Chen' -> 'Dr. Chen'."""
    if state.pcp is None:
        return "your doctor"
    parts = state.pcp.name.replace("Dr.", "").split()
    return f"Dr. {parts[-1]}" if parts else state.pcp.name


def pcp_office(state: CaseState, contacts: dict[str, str] | None = None) -> str:
    base = f"{pcp_short(state)}'s office" if state.pcp else "your doctor's office"
    contact = (contacts or {}).get(state.pcp.party_id if state.pcp else "pcp")
    return f"{base} ({contact})" if contact else base


def _owner_label(state: CaseState, track: Track, contacts: dict[str, str] | None) -> str:
    ts = state.track(track)
    owner = Owner(ts.owner)
    if owner == Owner.US or owner == Owner.NONE:
        return "our team"
    if owner == Owner.PATIENT:
        return "you"
    if owner == Owner.ADVOCATE:
        return "a nurse" if track == Track.PATIENT else "our team"
    if owner == Owner.PCP:
        return pcp_office(state, contacts)
    # supplier
    sid = ts.party_id or state.committed_sid or state.order_sent_to
    return supplier_name(sid) or "the equipment company"


def _pcp_due(state: CaseState) -> datetime | None:
    dues = [
        c.due_at
        for c in state.commitments.values()
        if c.status == CommitmentStatus.OPEN and c.party == (state.pcp.party_id if state.pcp else "pcp")
    ]
    return min(dues) if dues else None


# ----------------------------------------------------------------------------------------- items


def _order_item(state: CaseState, contacts: dict[str, str] | None) -> SnapshotItem:
    s = state.order.state
    doc = pcp_short(state)
    sup = supplier_name(state.order_sent_to or state.order_received_by or state.committed_sid) or (
        "the equipment company"
    )
    text = {
        OrderStatus.NOT_SENT: f"We're about to ask {doc}'s office for your signed wheelchair order.",
        OrderStatus.REQUESTED: f"{doc}'s office is preparing your signed wheelchair order and visit notes.",
        OrderStatus.DOCS_CHECK: f"We received paperwork from {doc}'s office and are checking it "
        "for Medicare.",
        OrderStatus.ADDENDUM_REQUESTED: f"{doc}'s office is adding a short note to your records that "
        "Medicare needs.",
        OrderStatus.SIGNED: f"{doc} signed your wheelchair order. We're double-checking it.",
        OrderStatus.CORRECTION_REQUESTED: f"{doc}'s office is fixing a small detail on your order.",
        OrderStatus.VALID: "Your wheelchair order is complete and checked.",
        OrderStatus.SENT_TO: f"Your order was sent to {sup}. We're confirming they received it.",
        OrderStatus.RECEIVED_BY: f"{sup} has your signed order.",
        OrderStatus.BLOCKED_ON_HUMAN: "Someone on our team is looking into your order personally.",
    }.get(s, "We're working on your wheelchair order.")
    due = state.order.due_at
    if s in (OrderStatus.REQUESTED, OrderStatus.ADDENDUM_REQUESTED, OrderStatus.CORRECTION_REQUESTED):
        due = _pcp_due(state) or due
    elif state.order.owner in (Owner.US, Owner.ADVOCATE):
        due = None
    return SnapshotItem(
        track=Track.ORDER,
        plain_status=text,
        owner_label=_owner_label(state, Track.ORDER, contacts),
        due_label=f"by {day_label(due)}" if due else None,
    )


def _supplier_item(state: CaseState, contacts: dict[str, str] | None) -> SnapshotItem:
    s = state.supplier.state
    sup = supplier_name(state.committed_sid)
    has_address = bool(state.patient_info and state.patient_info.address)
    text = {
        SupplierTrackStatus.NOT_STARTED: (
            "We'll start calling nearby equipment companies shortly."
            if has_address
            else "Once we have your address, we'll find a nearby equipment company."
        ),
        SupplierTrackStatus.CALLING: "We're calling nearby equipment companies to find one that has your "
        "wheelchair in stock and works with Medicare.",
        SupplierTrackStatus.QUALIFIED: "We found an equipment company that can help and are lining it up.",
        SupplierTrackStatus.COMMITTED: f"{sup or 'An equipment company'} will provide your wheelchair.",
        SupplierTrackStatus.EXHAUSTED: "Someone on our team is personally finding an equipment "
        "company for you.",
    }.get(s, "We're arranging an equipment company for you.")
    owner = _owner_label(state, Track.SUPPLIER, contacts)
    if s == SupplierTrackStatus.COMMITTED and sup:
        owner = sup
    return SnapshotItem(track=Track.SUPPLIER, plain_status=text, owner_label=owner, due_label=None)


def _assignment_item(state: CaseState, contacts: dict[str, str] | None) -> SnapshotItem:
    s = state.assignment.state
    sup = supplier_name(state.committed_sid) or "The equipment company"
    text = {
        AssignmentStatus.UNKNOWN: "We're confirming the equipment company accepts Medicare's approved "
        "price for every rental month.",
        AssignmentStatus.CONFIRMED_ALL_MONTHS: f"{sup} accepts Medicare's approved price for every "
        "rental month.",
        AssignmentStatus.REFUSED: "One company wouldn't accept Medicare's approved price, so we're "
        "using another.",
    }.get(s, "We're checking how the equipment company bills Medicare.")
    return SnapshotItem(
        track=Track.ASSIGNMENT,
        plain_status=text,
        owner_label=_owner_label(state, Track.ASSIGNMENT, contacts),
        due_label=None,
    )


def _slot_text(slot: datetime) -> str:
    return f"{day_label(slot)} at {time_label(slot)}"


def _delivery_item(state: CaseState, contacts: dict[str, str] | None) -> SnapshotItem:
    s = state.delivery.state
    sup = supplier_name(state.delivery.party_id or state.committed_sid) or "The equipment company"
    slot = state.delivery_slot
    due_label = None
    if s == DeliveryStatus.SCHEDULING and slot:
        text = f"{sup} can deliver your wheelchair on {_slot_text(slot)}. We need your OK."
        due_label = f"on {_slot_text(slot)}"
    elif s == DeliveryStatus.SCHEDULING:
        text = "We're arranging a delivery time with the equipment company."
    elif s == DeliveryStatus.SCHEDULED and slot:
        text = f"{sup} will deliver your wheelchair on {_slot_text(slot)}."
        due_label = f"on {_slot_text(slot)}"
    else:
        text = {
            DeliveryStatus.NOT_STARTED: "We'll set up a delivery time with you once the order is ready.",
            DeliveryStatus.SCHEDULED: "Your delivery is scheduled.",
            DeliveryStatus.DELIVERED: "Your wheelchair was delivered. Please tell us it's working for you.",
            DeliveryStatus.POD_CONFIRMED: "You confirmed your wheelchair arrived.",
        }.get(s, "We're working on your delivery.")
    owner = _owner_label(state, Track.DELIVERY, contacts)
    if s == DeliveryStatus.SCHEDULED and owner == "the equipment company":
        owner = sup
    return SnapshotItem(track=Track.DELIVERY, plain_status=text, owner_label=owner, due_label=due_label)


def _patient_item(state: CaseState, contacts: dict[str, str] | None) -> SnapshotItem:
    s = state.patient.state
    if s == PatientStatus.INTAKE:
        need = []
        if "address" not in state.patient_flags:
            need.append("your home address")
        if "consent" not in state.patient_flags:
            need.append("your OK for us to arrange the wheelchair for you")
        text = "We need " + " and ".join(need) + "." if need else "Thanks, we have what we need from you."
    else:
        text = {
            PatientStatus.CONSENTED: "Thanks, you've given us the go-ahead to arrange your wheelchair.",
            PatientStatus.COST_DISCLOSED: "We explained your cost and need to hear whether it works for you.",
            PatientStatus.COST_ACCEPTED: "You've said the cost works for you.",
            PatientStatus.CONCERN_RAISED: "A nurse will contact you about your question.",
            PatientStatus.DELIVERY_CONFIRMED: "You confirmed your wheelchair arrived and works.",
        }.get(s, "We're in touch with you about your wheelchair.")
    return SnapshotItem(
        track=Track.PATIENT,
        plain_status=text,
        owner_label=_owner_label(state, Track.PATIENT, contacts),
        due_label=None,
    )


def _next_patient_action(state: CaseState) -> str | None:
    p = state.patient.state
    if p == PatientStatus.INTAKE:
        if "address" not in state.patient_flags:
            return "Please share your home address, including the ZIP code."
        if "consent" not in state.patient_flags:
            return "Please let us know it's OK for us to arrange your wheelchair for you."
    if p == PatientStatus.COST_DISCLOSED:
        return "Please let us know whether the cost works for you."
    if state.delivery.state == DeliveryStatus.SCHEDULING and state.delivery_slot:
        return f"Please let us know if {_slot_text(state.delivery_slot)} works for delivery."
    if state.delivery.state == DeliveryStatus.DELIVERED:
        return "Please let us know your wheelchair arrived and is working."
    return None


def _headline(state: CaseState, items: dict[Track, SnapshotItem], action: str | None) -> str:
    c = state.case.state
    if c == CaseStatus.DONE or state.delivery.state == DeliveryStatus.POD_CONFIRMED:
        return "Your wheelchair has been delivered and everything is complete."
    if c == CaseStatus.CANCELLED:
        return "Your wheelchair request has been stopped."
    if c == CaseStatus.PAUSED:
        return "We've paused your wheelchair request for now."
    if c == CaseStatus.COVERAGE_CHANGED:
        return "Your insurance changed, so our team is reviewing your wheelchair request."
    if state.patient.state == PatientStatus.CONCERN_RAISED:
        return "A nurse will contact you about your question."
    if state.delivery.state in (DeliveryStatus.SCHEDULED, DeliveryStatus.DELIVERED):
        return items[Track.DELIVERY].plain_status
    if action:
        return "We need a quick answer from you to keep things moving."
    order = items[Track.ORDER]
    if order.due_label:
        return f"{order.plain_status} They expect to finish {order.due_label}."
    return order.plain_status


# --------------------------------------------------------------------------------------- snapshot

COST_SENTENCE = (
    "With Original Medicare Part B, you may pay the first month or so yourself until your ${deductible} "
    "yearly deductible is met. After that you pay {pct}% of Medicare's approved monthly rental, for up to "
    "{months} months, and then the wheelchair is yours."
)


def cost_text(cost: CostInfo) -> str:
    return COST_SENTENCE.format(
        deductible=cost.deductible_usd, pct=cost.coinsurance_pct, months=cost.rental_months
    )


def build_snapshot(state: CaseState, now: datetime, contacts: dict[str, str] | None = None) -> Snapshot:
    """Plain-language projection of `state`. `contacts` optionally maps party_id -> contact first name
    (e.g. {"pcp": "Maria"}) for labels like "Dr. Chen's office (Maria)"."""
    items = {
        Track.ORDER: _order_item(state, contacts),
        Track.SUPPLIER: _supplier_item(state, contacts),
        Track.ASSIGNMENT: _assignment_item(state, contacts),
        Track.DELIVERY: _delivery_item(state, contacts),
        Track.PATIENT: _patient_item(state, contacts),
    }
    action = _next_patient_action(state)
    headline = _headline(state, items, action)
    cost = CostInfo()  # constants only

    names: list[str] = []

    def add_name(n: str | None) -> None:
        if n and n not in names:
            names.append(n)

    if state.patient_info:
        add_name(state.patient_info.name)
        add_name(state.patient_info.name.split()[0])
        add_name(state.patient_info.profile.caregiver_name)
    if state.pcp:
        add_name(state.pcp.name)
        add_name(pcp_short(state))
        add_name(state.pcp.practice)
    for c in (contacts or {}).values():
        add_name(c)
    for sid in (state.committed_sid, state.order_sent_to, state.order_received_by, state.delivery.party_id):
        add_name(supplier_name(sid))

    dates: list[str] = []
    due_dts: list[datetime] = []
    o_due = state.order.due_at
    if items[Track.ORDER].due_label:
        o_due = _pcp_due(state) or o_due
        if o_due:
            due_dts.append(o_due)
    if items[Track.DELIVERY].due_label and state.delivery_slot:
        due_dts.append(state.delivery_slot)
    for dt in due_dts:
        for v in date_variants(dt, now):
            if v not in dates:
                dates.append(v)

    amounts = [
        f"${cost.deductible_usd}",
        f"{cost.coinsurance_pct}%",
        f"{cost.coinsurance_pct} percent",
        f"{cost.rental_months} months",
    ]
    return Snapshot(
        case_id=state.case_id,
        as_of=now,
        headline=headline,
        items=[items[t] for t in SNAPSHOT_TRACKS],
        cost=cost,
        next_patient_action=action,
        allowed_names=names,
        allowed_dates=dates,
        allowed_amounts=amounts,
    )


def snapshot_text(snapshot: Snapshot) -> str:
    """Canonical text handed to the LLM (the companion's only fact source)."""
    lines = ["CASE SNAPSHOT", f"Headline: {snapshot.headline}", "Status:"]
    for it in snapshot.items:
        line = f"- {it.track.value.title()}: {it.plain_status} Who has it: {it.owner_label}."
        if it.due_label:
            line += f" When: {it.due_label}."
        lines.append(line)
    lines.append(f"Cost: {cost_text(snapshot.cost)}")
    lines.append(f"What we need from the patient: {snapshot.next_patient_action or 'Nothing right now.'}")
    lines.append(f"Names you may use: {', '.join(snapshot.allowed_names) or 'none'}")
    lines.append(f"Dates you may use: {', '.join(snapshot.allowed_dates) or 'none'}")
    lines.append(f"Amounts you may use: {', '.join(snapshot.allowed_amounts)}")
    return "\n".join(lines)
