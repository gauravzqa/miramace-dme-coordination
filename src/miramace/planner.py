"""B2 — pure planner: plan(state, now, policy) -> PlannerDecision(tasks, events).

Returns the DESIRED set of open tasks (store.reconcile_tasks inserts new idem keys, cancels PENDING ones no
longer desired, never touches DONE/FAILED, and must NOT move due_at of an existing key) plus rule-derived
events (Source kind RULE). Deterministic: same (state, now, policy) → equal output, including event ids.
Nothing is planned when the case is not ACTIVE; nothing for a track whose owner is `advocate`.

Rules (plan-core "Planner"), evaluated in order; rule 12 (lease) is checked before 7–11 so a release in this
decision suppresses supplier/order/delivery tasks until the next plan:
  1 welcome_intake ladder   2 request_order   3 PCP follow-ups 8/16/24bh → ORDER_STALLED
  4 LCD addendum → LCD_MISSING_AFTER_ADDENDUM   5 CODE_DRIFT / correction → SWO_INCOMPLETE_AFTER_CORRECTION,
    ORDER_VALIDATED   6 rank + screen (one in flight, stop at 2 qualified) → SUPPLIERS_EXHAUSTED
  7 commit   8 cost_disclosure   9 send_order_<sid>   10 verify_order_received   11 reverify / propose_slot /
  delivery_slot notify   12 lease end / ≥2 unanswered reverify|verify → SUPPLIER_RELEASED (+ORDER_REDIRECT)
  13 verify_delivered
  14 delivered_checkin, CASE_DONE.

Task conventions (party_id, intent, version, args) — executor/interpret code against these:
  CALL_PCP (party "pcp"):
    request_order            v = order_request_version + 1        args {request_version}
    followup_order_{k}       target = earliest OPEN PCP commitment, else the latest PCP commitment if BROKEN;
                             k = max(followups + 1, rungs whose time has passed) (1..3; a 4th virtual rung at
                             due+32bh means "silent throughout" → ORDER_STALLED, as does followups ≥ 3);
                             v = 100*(order_request_version-1) + 1-based index of the commitment among PCP
                             commitments (dict order); due = add_business_hours(c.due_at, 24/48/72)
                             args {cid, what, k}
                             CONTRACT (interpret): fresh cid per request/resend; unanswered follow-up or one
                             with no new promise emits COMMITMENT_BROKEN(cid) (bumps followups); a new promise
                             re-makes the SAME cid (followups preserved) with the new due.
    request_addendum         v = addendum_requests + 1            args {missing: [criterion], wording: [str]}
    request_correction       v = correction_requests + 1          args {fields: [str]}
    send_order_{sid}         v = 1                                args {to_sid}
  CALL_SUPPLIER (party = sid):
    screen                   v = attempts + 1; NO PHI in args     args {zip, item, payer, rank, distance_mi}
    commit                   v = 1; PHI allowed later by executor args {zip, lease_h}
    verify_order_received    v = 1 + failures since commit (SupplierRecord.attempts) args {}
    reverify                 v = 10*m + 2*f (lease renewal, due lease_end - lead + f h), 10*m + 2*f + 1
                             (pre-scheduling freshness, due now), 10*m + 9 (lease-end wake-up timer, due
                             lease_end: the loop wakes there and rule 12 releases), m = minutes from SIM_START
                             to verified_at, f = unanswered reverify/verify calls since commit (reducer counts
                             SUPPLIER_NO_ANSWER on a COMMITTED record); f ≥ 2 → rule 12 releases args {reason}
    propose_slot             v = slot_proposals + 1 (re-proposed after decline/miss/lapsed slot)  args {}
    verify_delivered         v = 1 + minutes-stamp of the slot                                 args {slot}
  NOTIFY_PATIENT (party "patient"; intent = NotifyPurpose):
    welcome_intake / cost_disclosure   v = n + 1, n = notify_counts[purpose] (sends so far, answered or not:
                                       a re-ask always gets a fresh key); n < 3 to patient, n = 3 to caregiver
                                       (if a caregiver phone is known), then PATIENT_UNREACHABLE (a resolved
                                       escalation restarts the ladder)  args {purpose, to}
                                       welcome is re-asked while INTAKE (e.g. address given, consent missing);
                                       cost_disclosure is re-sent in COST_DISCLOSED at patient.due_at.
    delivery_slot            v = 1000*slot_proposals + n + 1 (n reset per slot)  args {purpose, to, slot}
    delivered_checkin        v = 1; capped to 1 notify per sim day                    args {purpose, to}

Assumptions on reducer semantics (B1): `notify_counts[purpose]` = sends per purpose; `last_notified_at`
updated on every PATIENT_NOTIFIED; `Commitment.followups` bumped when
a follow-up call for that commitment completes; ORDER_REQUEST_VERIFIED(received=False) returns order to
NOT_SENT so rule 2 re-sends with a bumped version; `assignment_by_sid` filled by ASSIGNMENT_CONFIRMED.
Escalation de-dup: a trigger is never re-opened while one is OPEN, nor re-opened for the same situation
(`context["key"]`) after resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from functools import lru_cache
from typing import Any

from miramace import rules
from miramace.clock import SIM_START, add_business_hours, next_business_time, next_patient_time
from miramace.escalation import open_escalation
from miramace.events import Event, EventType, PlannerDecision, Source, make_event
from miramace.geo import distance_mi
from miramace.models import (
    CHICAGO,
    PCP_PARTY_ID,
    CaseState,
    CaseStatus,
    CommitmentStatus,
    DeliveryStatus,
    EscalationStatus,
    EscalationTrigger,
    OrderStatus,
    Owner,
    PatientStatus,
    Supplier,
    SupplierRecord,
    SupplierStatus,
    TaskKind,
    TaskSpec,
    Track,
    idem_key,
    load_suppliers,
)

PATIENT_PARTY_ID = "patient"
SCREEN_ITEM = "K0001 standard manual wheelchair (rental)"
SCREEN_PAYER = "Original Medicare Part B"


@dataclass(frozen=True)
class Policy:
    screen_attempts: int = 3  # per supplier
    screen_retry_bh: float = 2.0
    qualified_target: int = 2  # primary + backup, then screening stops
    # with a usable supplier in hand, stop hunting a backup once this many suppliers were contacted
    backup_screen_cap: int = 6
    # next business morning, then daily
    pcp_followup_offsets_bh: tuple[float, ...] = (8.0, 16.0, 24.0)
    call_max_attempts: int = 3
    reverify_attempts: int = 2
    reverify_lead_h: float = 2.0  # lease reverify is due this long before the lease ends
    lease_h: float = 72.0
    fresh_h: float = 24.0  # verification freshness required to schedule
    patient_attempts: int = 3  # in best hours, then caregiver, then escalate
    patient_retry_h: float = 4.0
    notify_cap_per_day: int = 1
    action_required: frozenset[str] = field(
        default_factory=lambda: frozenset({"welcome_intake", "cost_disclosure", "delivery_slot"})
    )


_CONSENTED_STATES = {
    PatientStatus.CONSENTED,
    PatientStatus.COST_DISCLOSED,
    PatientStatus.COST_ACCEPTED,
    PatientStatus.DELIVERY_CONFIRMED,
}
_FOLLOWUP_ORDER_STATES = {
    OrderStatus.REQUESTED,
    OrderStatus.DOCS_CHECK,
    OrderStatus.ADDENDUM_REQUESTED,
    OrderStatus.SIGNED,
    OrderStatus.CORRECTION_REQUESTED,
    OrderStatus.VALID,
    OrderStatus.SENT_TO,
}
_PRE_DELIVERY = {DeliveryStatus.NOT_STARTED, DeliveryStatus.SCHEDULING}


@lru_cache(maxsize=1)
def _directory() -> dict[str, Supplier]:
    return {s.sid: s for s in load_suppliers()}


def _stamp(dt: datetime | None) -> int:
    """Minutes from SIM_START (≥ 0) — used to give per-epoch task versions a stable int."""
    if dt is None:
        return 0
    return max(0, int((dt - SIM_START).total_seconds() // 60))


@dataclass
class _Ctx:
    state: CaseState
    now: datetime
    policy: Policy
    tasks: list[tuple[Track, TaskSpec]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    suppress: set[Track] = field(default_factory=set)

    # ---- helpers
    def owner(self, track: Track) -> Owner:
        return Owner(self.state.track(track).owner)

    def blocked(self, *tracks: Track) -> bool:
        return any(self.owner(t) == Owner.ADVOCATE or t in self.suppress for t in tracks)

    def task(
        self,
        track: Track,
        kind: TaskKind,
        party_id: str,
        intent: str,
        version: int,
        due: datetime,
        args: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if self.blocked(track):
            return
        spec = TaskSpec(
            kind=kind,
            party_id=party_id,
            intent=intent,
            version=version,
            idem_key=idem_key(self.state.case_id, party_id, intent, version),
            due_at=due,
            max_attempts=max_attempts or self.policy.call_max_attempts,
            args=args or {},
        )
        self.tasks.append((track, spec))

    def call_due(self, base: datetime | None = None) -> datetime:
        return next_business_time(max(self.now, base) if base else self.now)

    def event(self, type_: EventType, payload: dict[str, Any] | None, rule: str) -> None:
        ev = make_event(self.state.case_id, type_, payload, Source(kind="RULE", ref=rule), self.now)
        eid = f"{self.state.case_id}:rule:{self.state.last_seq}:{type_.value.lower()}:{len(self.events)}"
        self.events.append(ev.model_copy(update={"event_id": eid}))

    def escalate(self, trigger: EscalationTrigger, key: str, context: dict[str, Any]) -> bool:
        """Open an escalation unless one is OPEN for this trigger or one (any status) has the same key."""
        for e in self.state.escalations.values():
            if e.trigger == trigger and (e.status == EscalationStatus.OPEN or e.context.get("key") == key):
                return False
        ev = open_escalation(
            trigger,
            self.state,
            self.now,
            {**context, "key": key},
            source=Source(kind="RULE", ref=f"planner:{trigger.value}"),
        )
        self.events.append(ev)
        self.suppress.add(Track(ev.payload["escalation"]["track"]))
        return True


# ------------------------------------------------------------------------------------------ helpers


def _consented(s: CaseState) -> bool:
    return PatientStatus(s.patient.state) in _CONSENTED_STATES or {"address", "consent"} <= s.patient_flags


def _patient_zip(s: CaseState) -> str | None:
    if s.patient_info and s.patient_info.address:
        return s.patient_info.address.zip
    return None


def _ranking(s: CaseState) -> list[tuple[str, float]]:
    """(sid, miles) in rank order: state.ranked if present, else computed from the directory."""
    pz = _patient_zip(s)
    directory = _directory()
    if s.ranked:
        out = []
        for sid in s.ranked:
            sup = directory.get(sid)
            miles = distance_mi(pz, sup.zip) if (pz and sup) else float("nan")
            out.append((sid, miles))
        return out
    if pz is None:
        return []
    return rules.rank_suppliers(list(directory.values()), pz)


def _rec(s: CaseState, sid: str) -> SupplierRecord:
    return s.suppliers.get(sid) or SupplierRecord(sid=sid)


def _lease_end(rec: SupplierRecord, p: Policy) -> datetime | None:
    candidates = [t for t in (rec.expires_at,) if t is not None]
    if rec.verified_at is not None:
        candidates.append(rec.verified_at + timedelta(hours=p.lease_h))
    return max(candidates) if candidates else None


def _best_backup(s: CaseState, exclude: str) -> str | None:
    for sid in s.ranked or [sid for sid, _ in _ranking(s)]:
        rec = s.suppliers.get(sid)
        if sid != exclude and rec and rec.status == SupplierStatus.QUALIFIED and rules.assignment_ok(sid, s):
            return sid
    return None


def _patient_notify(
    c: _Ctx,
    purpose: str,
    track: Track,
    version_base: int = 0,
    extra: dict[str, Any] | None = None,
    not_before: datetime | None = None,
) -> None:
    """Action-required notify with the ladder: 3 sends → caregiver → PATIENT_UNREACHABLE.

    n = sends of this purpose so far (answered or not), so an answered-but-incomplete reply still yields a new
    idem key for the re-ask. Sends already counted by a RESOLVED unreachable escalation do not count again."""
    s, p = c.state, c.policy
    if c.blocked(Track.PATIENT, track):
        return
    profile = s.patient_info.profile if s.patient_info else None
    n = s.notify_counts.get(purpose, 0)
    tag = f"{purpose}:{version_base}"
    spent = max(
        (
            int(e.context.get("sends", 0))
            for e in s.escalations.values()
            if e.trigger == EscalationTrigger.PATIENT_UNREACHABLE
            and e.status == EscalationStatus.RESOLVED
            and e.context.get("ladder") == tag
        ),
        default=0,
    )
    m = max(0, n - spent)  # sends in the current ladder
    caregiver = bool(profile and profile.caregiver_phone)
    if m < p.patient_attempts:
        to = PATIENT_PARTY_ID
    elif m == p.patient_attempts and caregiver:
        to = "caregiver"
    else:
        c.escalate(
            EscalationTrigger.PATIENT_UNREACHABLE,
            key=f"{tag}:{n}",
            context={"purpose": purpose.replace("_", " "), "attempts": m, "ladder": tag, "sends": n},
        )
        return
    base = c.now
    if m > 0 and s.last_notified_at is not None:
        base = max(c.now, s.last_notified_at + timedelta(hours=p.patient_retry_h))
    if not_before is not None:
        base = max(base, not_before)
    due = next_patient_time(base, profile)
    args = {"purpose": purpose, "to": to, **(extra or {})}
    c.task(track, TaskKind.NOTIFY_PATIENT, PATIENT_PARTY_ID, purpose, version_base + n + 1, due, args)


def _capped_notify(c: _Ctx, purpose: str, track: Track, version: int) -> None:
    """Informational notify: at most notify_cap_per_day (1) per sim day → pushed to the next day's window."""
    s = c.state
    if c.blocked(Track.PATIENT, track):
        return
    profile = s.patient_info.profile if s.patient_info else None
    base = c.now
    if (
        s.last_notified_at is not None
        and s.last_notified_at.astimezone(CHICAGO).date() == c.now.astimezone(CHICAGO).date()
    ):
        tomorrow = c.now.astimezone(CHICAGO).date() + timedelta(days=1)
        base = datetime.combine(tomorrow, time(0), tzinfo=CHICAGO)
    due = next_patient_time(base, profile)
    c.task(
        track,
        TaskKind.NOTIFY_PATIENT,
        PATIENT_PARTY_ID,
        purpose,
        version,
        due,
        {"purpose": purpose, "to": PATIENT_PARTY_ID},
    )


# ------------------------------------------------------------------------------------------ rules


def _r1_welcome(c: _Ctx) -> None:
    if PatientStatus(c.state.patient.state) == PatientStatus.INTAKE:
        _patient_notify(c, "welcome_intake", Track.PATIENT)


def _r2_request_order(c: _Ctx) -> None:
    s = c.state
    if _consented(s) and OrderStatus(s.order.state) == OrderStatus.NOT_SENT:
        v = s.order_request_version + 1
        c.task(
            Track.ORDER,
            TaskKind.CALL_PCP,
            PCP_PARTY_ID,
            "request_order",
            v,
            c.call_due(),
            {"request_version": v},
        )


def _followup_rungs(due_at: datetime, offs: tuple[float, ...]) -> list[datetime]:
    """Follow-up times after a PCP promise. A promise due mid-business-day gets due+offs; one due at/after
    close ("by Thursday" = Thu 17:00) is anchored on the next business morning so the first rung is that
    morning (plain business-hour arithmetic would land a full business day later)."""
    anchor = next_business_time(due_at)
    if anchor == due_at:
        return [add_business_hours(due_at, o) for o in offs]
    return [add_business_hours(anchor, o - offs[0]) for o in offs]


def _r3_pcp_followups(c: _Ctx) -> None:
    s, p = c.state, c.policy
    if OrderStatus(s.order.state) not in _FOLLOWUP_ORDER_STATES:
        return
    pcp = [cm for cm in s.commitments.values() if cm.party == PCP_PARTY_ID]
    open_ = [cm for cm in pcp if cm.status == CommitmentStatus.OPEN]
    if open_:
        cm = min(open_, key=lambda x: (x.due_at, x.cid))
    elif pcp and pcp[-1].status == CommitmentStatus.BROKEN:
        cm = pcp[-1]  # promise broken (unanswered follow-up / no new date): keep climbing the ladder
    else:
        return
    offs = p.pcp_followup_offsets_bh
    rungs = _followup_rungs(cm.due_at, (*offs, offs[-1] + offs[0]))
    k = max(cm.followups + 1, sum(1 for t in rungs if t <= c.now))
    if k > len(offs):
        c.escalate(
            EscalationTrigger.ORDER_STALLED,
            key=cm.cid,
            context={"cid": cm.cid, "what": cm.what, "followups": max(cm.followups, len(offs))},
        )
        return
    epoch = max(0, s.order_request_version - 1)
    version = 100 * epoch + 1 + [x.cid for x in pcp].index(cm.cid)
    due = rungs[k - 1]
    c.task(
        Track.ORDER,
        TaskKind.CALL_PCP,
        PCP_PARTY_ID,
        f"followup_order_{k}",
        version,
        due,
        {"cid": cm.cid, "what": cm.what, "k": k},
    )


def _r4_lcd(c: _Ctx) -> None:
    s = c.state
    order = OrderStatus(s.order.state)
    if s.lcd is None or order not in (OrderStatus.DOCS_CHECK, OrderStatus.SIGNED):
        return
    missing = s.lcd.missing
    if not missing:
        return
    if s.addendum_requests > 0 and len(s.visit_notes) <= s.addendum_notes_seen:
        return  # addendum requested: wait for a note received AFTER it (PCP follow-ups cover silence)
    if s.addendum_requests == 0:
        c.task(
            Track.ORDER,
            TaskKind.CALL_PCP,
            PCP_PARTY_ID,
            "request_addendum",
            1,
            c.call_due(),
            {
                "missing": [m.value for m in missing],
                "wording": [rules.lcd_missing_wording(m) for m in missing],
            },
        )
    else:
        c.escalate(
            EscalationTrigger.LCD_MISSING_AFTER_ADDENDUM,
            key=f"notes:{len(s.visit_notes)}:{','.join(m.value for m in missing)}",
            context={"missing": [m.value for m in missing], "addendum_requests": s.addendum_requests},
        )


def _r5_swo(c: _Ctx) -> None:
    s = c.state
    if OrderStatus(s.order.state) != OrderStatus.SIGNED or s.swo is None or c.blocked(Track.ORDER):
        return
    code = rules.check_code(s.swo.hcpcs)
    sig = f"{s.swo.signature_date}:{s.correction_requests}"
    if code == "DRIFT":
        c.escalate(EscalationTrigger.CODE_DRIFT, key=f"{s.swo.hcpcs}:{sig}", context={"hcpcs": s.swo.hcpcs})
        return
    missing = rules.swo_completeness(s.swo)
    if missing:
        if s.correction_requests == 0:
            c.task(
                Track.ORDER,
                TaskKind.CALL_PCP,
                PCP_PARTY_ID,
                "request_correction",
                1,
                c.call_due(),
                {"fields": missing},
            )
        else:
            c.escalate(
                EscalationTrigger.SWO_INCOMPLETE_AFTER_CORRECTION,
                key=f"{','.join(missing)}:{sig}",
                context={"missing": missing, "correction_requests": s.correction_requests},
            )
        return
    if s.lcd is not None and s.lcd.all_met:
        c.event(EventType.ORDER_VALIDATED, None, "planner:order_validated")


def _backup_hunt_over(s: CaseState, p: Policy, ranking: list[tuple[str, float]]) -> bool:
    """No more cold screening calls for a BACKUP once a usable supplier exists and the case is past the point
    where a backup helps: delivery scheduled (or later); the order is validated / sent / received (the order
    goes to the supplier we have — a ghost there is handled by release + redirect, and screening resumes only
    if no usable supplier is left); or `backup_screen_cap` suppliers have been contacted. Never stops a hunt
    with no usable supplier, so SUPPLIERS_EXHAUSTED behaviour is unchanged."""
    usable = [
        sid
        for sid, _ in ranking
        if _rec(s, sid).status == SupplierStatus.COMMITTED
        or (_rec(s, sid).status == SupplierStatus.QUALIFIED and rules.assignment_ok(sid, s))
    ]
    if not usable:
        return False
    if DeliveryStatus(s.delivery.state) not in _PRE_DELIVERY:
        return True
    if OrderStatus(s.order.state) in (OrderStatus.VALID, OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY):
        return True
    contacted = sum(1 for sid, _ in ranking if _rec(s, sid).status != SupplierStatus.UNCONTACTED)
    return contacted >= p.backup_screen_cap


def _r6_screen(c: _Ctx) -> None:
    s, p = c.state, c.policy
    pz = _patient_zip(s)
    if pz is None or c.blocked(Track.SUPPLIER):
        return
    ranking = _ranking(s)
    if not s.ranked:
        c.event(
            EventType.CANDIDATES_RANKED,
            {"sids": [sid for sid, _ in ranking], "distances": dict(ranking)},
            "planner:rank_suppliers",
        )
    good = [
        sid
        for sid, _ in ranking
        if _rec(s, sid).status in (SupplierStatus.QUALIFIED, SupplierStatus.COMMITTED)
    ]
    if len(good) >= p.qualified_target or _backup_hunt_over(s, p, ranking):
        return
    fresh = [
        (i, sid, d) for i, (sid, d) in enumerate(ranking) if _rec(s, sid).status == SupplierStatus.UNCONTACTED
    ]
    retry = [
        (i, sid, d)
        for i, (sid, d) in enumerate(ranking)
        if _rec(s, sid).status in (SupplierStatus.NO_ANSWER, SupplierStatus.CALLING)
        and _rec(s, sid).attempts < p.screen_attempts
    ]
    pick = (fresh or retry or [None])[0]
    if pick is None:
        if not good:
            statuses = sorted({_rec(s, sid).status.value for sid, _ in ranking})
            c.escalate(
                EscalationTrigger.SUPPLIERS_EXHAUSTED,
                key=f"ranked:{len(ranking)}",
                context={"count": len(ranking), "summary": "; ".join(statuses)},
            )
        return
    i, sid, miles = pick
    rec = _rec(s, sid)
    due = c.call_due() if rec.attempts == 0 else add_business_hours(c.now, p.screen_retry_bh)
    args = {"zip": pz, "item": SCREEN_ITEM, "payer": SCREEN_PAYER, "rank": i + 1, "distance_mi": miles}
    c.task(Track.SUPPLIER, TaskKind.CALL_SUPPLIER, sid, "screen", rec.attempts + 1, due, args, max_attempts=1)


def _r7_commit(c: _Ctx) -> None:
    s, p = c.state, c.policy
    if s.committed_sid or not _consented(s):
        return
    if any(r.status == SupplierStatus.COMMITTED for r in s.suppliers.values()):
        return
    for sid, _ in _ranking(s):
        rec = s.suppliers.get(sid)
        if rec and rec.status == SupplierStatus.QUALIFIED and rules.assignment_ok(sid, s):
            c.task(
                Track.SUPPLIER,
                TaskKind.CALL_SUPPLIER,
                sid,
                "commit",
                1,
                c.call_due(),
                {"zip": _patient_zip(s), "lease_h": p.lease_h},
            )
            return


def _r8_cost(c: _Ctx) -> None:
    s = c.state
    st = PatientStatus(s.patient.state)
    if s.committed_sid and st == PatientStatus.CONSENTED:
        _patient_notify(c, "cost_disclosure", Track.PATIENT)
    elif st == PatientStatus.COST_DISCLOSED:
        # no decision yet: re-send at the decision due time (each send re-stamps due_at); ladder → unreachable
        _patient_notify(c, "cost_disclosure", Track.PATIENT, not_before=s.patient.due_at or c.now)


def _r9_send_order(c: _Ctx) -> None:
    s = c.state
    if OrderStatus(s.order.state) == OrderStatus.VALID and s.committed_sid:
        sid = s.committed_sid
        c.task(
            Track.ORDER,
            TaskKind.CALL_PCP,
            PCP_PARTY_ID,
            f"send_order_{sid}",
            1,
            c.call_due(),
            {"to_sid": sid},
        )


def _r10_verify_received(c: _Ctx) -> None:
    s = c.state
    if OrderStatus(s.order.state) != OrderStatus.SENT_TO or not s.order_sent_to:
        return
    open_pcp = [
        cm.due_at
        for cm in s.commitments.values()
        if cm.party == PCP_PARTY_ID and cm.status == CommitmentStatus.OPEN
    ]
    base = min(open_pcp) if open_pcp else (s.order.due_at or c.now)
    f = _rec(s, s.order_sent_to).attempts  # unanswered verify/reverify calls since commit
    due = c.call_due(base) if f == 0 else add_business_hours(max(c.now, base), c.policy.screen_retry_bh)
    c.task(Track.ORDER, TaskKind.CALL_SUPPLIER, s.order_sent_to, "verify_order_received", 1 + f, due)


def _r11_schedule(c: _Ctx) -> None:
    s, p = c.state, c.policy
    sid = s.committed_sid
    if not sid:
        return
    rec = _rec(s, sid)
    delivery = DeliveryStatus(s.delivery.state)
    received = OrderStatus(s.order.state) == OrderStatus.RECEIVED_BY and s.order_received_by == sid
    accepted = PatientStatus(s.patient.state) == PatientStatus.COST_ACCEPTED
    m, f = 10 * _stamp(rec.verified_at), rec.attempts  # f = unanswered calls since commit
    slot = s.delivery_slot
    lapsed = delivery == DeliveryStatus.SCHEDULING and slot is not None and slot <= c.now
    if delivery in _PRE_DELIVERY:
        if received and accepted:
            stale = rec.verified_at is None or c.now - rec.verified_at > timedelta(hours=p.fresh_h)
            if stale:  # e.g. slot confirmation rejected as stale: reverify, then re-ask / re-propose
                if f < p.reverify_attempts:
                    c.task(
                        Track.SUPPLIER,
                        TaskKind.CALL_SUPPLIER,
                        sid,
                        "reverify",
                        m + 2 * f + 1,
                        c.call_due(),
                        {"reason": "pre_schedule"},
                        max_attempts=p.reverify_attempts,
                    )
                return
            if delivery == DeliveryStatus.NOT_STARTED or slot is None or lapsed:
                v = s.slot_proposals + 1
                c.task(Track.DELIVERY, TaskKind.CALL_SUPPLIER, sid, "propose_slot", v, c.call_due())
        # lease renewal reverify: due shortly before the lease lapses (retry 1h later after a no-answer),
        # plus a wake-up at lease end so the loop reaches the time at which rule 12 releases a ghost.
        end = _lease_end(rec, p)
        if end is not None and not (received and accepted) and f < p.reverify_attempts:
            lead = end - timedelta(hours=p.reverify_lead_h) + timedelta(hours=f)
            for v, due in ((m + 2 * f, c.call_due(lead)), (m + 9, c.call_due(end))):
                c.task(
                    Track.SUPPLIER,
                    TaskKind.CALL_SUPPLIER,
                    sid,
                    "reverify",
                    v,
                    due,
                    {"reason": "lease"},
                    max_attempts=p.reverify_attempts,
                )
    if delivery == DeliveryStatus.SCHEDULING and slot is not None and not lapsed:
        _patient_notify(
            c,
            "delivery_slot",
            Track.DELIVERY,
            version_base=1000 * s.slot_proposals,
            extra={"slot": slot.isoformat()},
        )


def _r12_lease(c: _Ctx) -> None:
    s, p = c.state, c.policy
    sid = s.committed_sid
    if not sid or DeliveryStatus(s.delivery.state) not in _PRE_DELIVERY or c.blocked(Track.SUPPLIER):
        return
    rec = _rec(s, sid)
    end = _lease_end(rec, p)
    if rec.attempts >= p.reverify_attempts:
        checking_receipt = s.order_sent_to == sid and OrderStatus(s.order.state) == OrderStatus.SENT_TO
        what = "order-receipt checks" if checking_receipt else "re-verification calls"
        reason = f"no answer to {rec.attempts} {what}"
    elif end is None:
        return
    else:
        reverify_due = next_business_time(
            max(end - timedelta(hours=p.reverify_lead_h), rec.verified_at or end)
        )
        if c.now < end or c.now <= reverify_due:
            return
        reason = "lease expired without re-verification"
    c.event(EventType.SUPPLIER_RELEASED, {"sid": sid, "reason": reason}, "planner:lease_expired")
    order = OrderStatus(s.order.state)
    if s.order_sent_to == sid and order in (OrderStatus.SENT_TO, OrderStatus.RECEIVED_BY):
        backup = _best_backup(s, exclude=sid)
        if backup:
            c.event(EventType.ORDER_REDIRECT, {"from_sid": sid, "to_sid": backup}, "planner:lease_expired")
    c.suppress |= {Track.SUPPLIER, Track.ORDER, Track.DELIVERY}


def _r13_verify_delivered(c: _Ctx) -> None:
    s = c.state
    if DeliveryStatus(s.delivery.state) != DeliveryStatus.SCHEDULED or not s.committed_sid:
        return
    slot = s.delivery_slot
    due = s.delivery.due_at or (slot + timedelta(hours=4) if slot else c.now)
    c.task(
        Track.DELIVERY,
        TaskKind.CALL_SUPPLIER,
        s.committed_sid,
        "verify_delivered",
        1 + _stamp(slot),
        c.call_due(due),
        {"slot": slot.isoformat() if slot else None},
    )


def _r14_done(c: _Ctx) -> None:
    s = c.state
    delivery = DeliveryStatus(s.delivery.state)
    if delivery == DeliveryStatus.DELIVERED:
        _capped_notify(c, "delivered_checkin", Track.DELIVERY, 1)
    elif delivery == DeliveryStatus.POD_CONFIRMED:
        c.event(EventType.CASE_DONE, None, "planner:case_done")


# ------------------------------------------------------------------------------------------ entry


def plan(state: CaseState, now: datetime, policy: Policy = Policy()) -> PlannerDecision:  # noqa: B008
    """Desired open tasks + rule-derived events for this state at `now`. Pure and deterministic."""
    if CaseStatus(state.case.state) != CaseStatus.ACTIVE:
        return PlannerDecision()
    c = _Ctx(state=state, now=now, policy=policy)
    _r1_welcome(c)
    _r2_request_order(c)
    _r3_pcp_followups(c)
    _r4_lcd(c)
    _r5_swo(c)
    _r6_screen(c)
    _r12_lease(c)
    _r7_commit(c)
    _r8_cost(c)
    _r9_send_order(c)
    _r10_verify_received(c)
    _r11_schedule(c)
    _r13_verify_delivered(c)
    _r14_done(c)
    seen: set[str] = set()
    tasks: list[TaskSpec] = []
    for track, spec in c.tasks:
        if track in c.suppress or spec.idem_key in seen:
            continue
        seen.add(spec.idem_key)
        tasks.append(spec)
    tasks.sort(key=lambda t: (t.due_at, t.idem_key))
    return PlannerDecision(tasks=tasks, events=c.events)
