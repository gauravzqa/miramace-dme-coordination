"""B7 — rich rendering for the CLI: timeline rows, live printer, summaries, metrics, snapshot, transcripts,
escalations. Pure presentation: reads events/models, never mutates anything.

Timeline row = (sim time "Mon 10:05", track, event type, source chip, one-line summary [+ evidence quote]).
A dim header separates sim days so a multi-day run reads as a story.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from miramace.events import Event, EventType
from miramace.models import CHICAGO, PCP_PARTY_ID, Track

ET = EventType

# --------------------------------------------------------------------------------------------- tracks

TRACK_BY_TYPE: dict[EventType, str] = {
    **{
        t: "case"
        for t in (
            ET.CASE_OPENED,
            ET.CASE_PAUSED,
            ET.CASE_RESUMED,
            ET.CASE_CANCELLED,
            ET.COVERAGE_CHANGED,
            ET.CASE_DONE,
        )
    },
    **{
        t: "patient"
        for t in (
            ET.PATIENT_NOTIFIED,
            ET.ADDRESS_CONFIRMED,
            ET.CONSENT_GIVEN,
            ET.COST_DISCLOSED,
            ET.COST_ACCEPTED,
            ET.CONCERN_RAISED,
            ET.STATUS_ASKED,
            ET.PATIENT_UNREACHABLE,
        )
    },
    **{
        t: "order"
        for t in (
            ET.SWO_DRAFTED,
            ET.ORDER_REQUESTED,
            ET.ORDER_REQUEST_VERIFIED,
            ET.VISIT_NOTE_RECEIVED,
            ET.LCD_EVALUATED,
            ET.ADDENDUM_REQUESTED,
            ET.SWO_SIGNED_RECEIVED,
            ET.SWO_EVALUATED,
            ET.CORRECTION_REQUESTED,
            ET.ORDER_VALIDATED,
            ET.ORDER_SENT,
            ET.ORDER_RECEIVED,
            ET.ORDER_REDIRECT,
            ET.ORDER_STALLED,
            ET.PCP_REFUSED,
        )
    },
    **{
        t: "supplier"
        for t in (
            ET.CANDIDATES_RANKED,
            ET.SUPPLIER_NO_ANSWER,
            ET.SUPPLIER_QUALIFIED,
            ET.SUPPLIER_DISQUALIFIED,
            ET.SUPPLIER_COMMITTED,
            ET.SUPPLIER_REVERIFIED,
            ET.SUPPLIER_RELEASED,
            ET.SUPPLIERS_EXHAUSTED,
        )
    },
    ET.ASSIGNMENT_CONFIRMED: "assignment",
    ET.ASSIGNMENT_REFUSED: "assignment",
    **{
        t: "delivery"
        for t in (
            ET.DELIVERY_SLOT_PROPOSED,
            ET.DELIVERY_SCHEDULED,
            ET.DELIVERY_MISSED,
            ET.DELIVERED,
            ET.POD_CONFIRMED,
        )
    },
    **{t: "ledger" for t in (ET.COMMITMENT_MADE, ET.COMMITMENT_KEPT, ET.COMMITMENT_BROKEN)},
    ET.CALL_COMPLETED: "audit",
    ET.EXTRACTION_FAILED: "audit",
}

TRACK_STYLE = {
    "case": "bold white",
    "patient": "green",
    "order": "blue",
    "supplier": "cyan",
    "assignment": "cyan",
    "delivery": "magenta",
    "ledger": "yellow",
    "audit": "dim",
}

SOURCE_LABEL = {
    "TRANSCRIPT": "TRANSCRIPT",
    "RULE": "RULE",
    "HUMAN": "HUMAN",
    "PATIENT_INTENT": "PATIENT",
    "INBOUND_DOC": "INBOUND",
    "TIMER": "TIMER",
    "SYSTEM": "SYSTEM",
}
SOURCE_STYLE = {
    "TRANSCRIPT": "black on cyan",
    "RULE": "white on blue",
    "HUMAN": "black on magenta",
    "PATIENT": "black on green",
    "INBOUND": "black on yellow",
    "TIMER": "white on grey37",
    "SYSTEM": "white on grey23",
}

TYPE_STYLE: dict[EventType, str] = {
    ET.CASE_DONE: "bold green",
    ET.CASE_CANCELLED: "bold red",
    ET.ESCALATION_OPENED: "bold red",
    ET.HUMAN_RESOLVED: "bold magenta",
    ET.SUPPLIER_DISQUALIFIED: "red",
    ET.SUPPLIER_RELEASED: "red",
    ET.ORDER_REDIRECT: "bold yellow",
    ET.COMMITMENT_BROKEN: "red",
    ET.CONCERN_RAISED: "bold red",
    ET.SUPPLIER_COMMITTED: "bold cyan",
    ET.ORDER_VALIDATED: "bold blue",
    ET.DELIVERY_SCHEDULED: "bold magenta",
    ET.POD_CONFIRMED: "bold green",
}

# events that are noise in a demo timeline unless asked for (shown with --all / track filter "audit")
QUIET_TYPES = {ET.CALL_COMPLETED}

TIME_W, TRACK_W, TYPE_W, SRC_W = 10, 11, 24, 12


# ------------------------------------------------------------------------------------------- helpers


def _name(sid: str | None) -> str:
    if not sid:
        return "?"
    if sid == PCP_PARTY_ID:
        return "Dr. Chen's office"
    try:
        from miramace.snapshot import supplier_name

        return supplier_name(sid) or sid
    except Exception:  # pragma: no cover - defensive
        return sid


def _short(sid: str | None) -> str:
    """'Lakeshore Home Medical Equipment' -> 'Lakeshore'."""
    n = _name(sid)
    return n.split()[0] if n and n != "Dr. Chen's office" else n


def _ct(dt: datetime | str | None) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return None
    return dt.astimezone(CHICAGO) if dt.tzinfo else dt.replace(tzinfo=CHICAGO)


def sim_time_label(dt: datetime | str | None) -> str:
    d = _ct(dt)
    return d.strftime("%a %H:%M") if d else "--- --:--"


def day_header(dt: datetime | str | None) -> str:
    d = _ct(dt)
    return f"{d.strftime('%A')}, {d.strftime('%b')} {d.day}" if d else "?"


def _when(dt: Any) -> str:
    d = _ct(dt)
    return d.strftime("%a %b %-d %H:%M") if d else "?"


def _clip(s: str, n: int = 110) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _q(quote: str | None, n: int = 70) -> str:
    return f' "{_clip(quote, n)}"' if quote else ""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _fields(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------ summaries


def event_summary(e: Event) -> str:
    """One readable line per event. Evidence quotes are appended in quotes when the payload has one."""
    p = e.payload or {}
    t = e.type
    try:
        if t == ET.CASE_OPENED:
            pat = p.get("patient", {})
            return f"Case opened for {pat.get('name', 'patient')} (PCP {p.get('pcp', {}).get('name', '?')})"
        if t == ET.CASE_DONE:
            return "Case closed: chair delivered and confirmed"
        if t in (ET.CASE_CANCELLED, ET.CASE_PAUSED):
            return f"{t.value.split('_')[1].title()}: {p.get('reason')}"
        if t == ET.CASE_RESUMED:
            return "Case resumed"
        if t == ET.COVERAGE_CHANGED:
            return f"Coverage changed to {p.get('new')}"
        if t == ET.PATIENT_NOTIFIED:
            ans = "answered" if p.get("answered") else "no answer"
            ok = "" if p.get("grounding_ok", True) else " [grounding HOLD]"
            return f"Told Eleanor ({p.get('purpose')}, {ans}){ok}:{_q(p.get('text'), 60)}"
        if t == ET.ADDRESS_CONFIRMED:
            a = p.get("address", {})
            return f"Address confirmed: {a.get('line1')}, {a.get('zip')}"
        if t == ET.CONSENT_GIVEN:
            return f"Consent given ({p.get('scope')})"
        if t == ET.COST_DISCLOSED:
            c = p.get("cost", {})
            return (
                f"Cost explained: ${c.get('deductible_usd')} deductible, {c.get('coinsurance_pct')}% "
                f"for {c.get('rental_months')} rental months"
            )
        if t == ET.COST_ACCEPTED:
            return "Eleanor accepted the cost"
        if t == ET.CONCERN_RAISED:
            return f"Concern ({p.get('kind')}):{_q(p.get('utterance'))}"
        if t == ET.STATUS_ASKED:
            return f"Eleanor asked:{_q(p.get('utterance'))}"
        if t == ET.PATIENT_UNREACHABLE:
            return "Eleanor unreachable after attempts"
        if t == ET.SWO_DRAFTED:
            return f"SWO drafted ({p.get('swo', {}).get('hcpcs')})"
        if t == ET.ORDER_REQUESTED:
            return f"Order requested from Dr. Chen (ref {p.get('request_ref')}, v{p.get('version')})"
        if t == ET.ORDER_REQUEST_VERIFIED:
            if p.get("received"):
                owner = f", owner {p['named_owner']}" if p.get("named_owner") else ""
                return f"PCP confirmed request received{owner}"
            return "PCP says request NOT received -> resend"
        if t == ET.VISIT_NOTE_RECEIVED:
            n = p.get("note", {})
            kind = "Addendum" if n.get("is_addendum") else "Visit note"
            return f"{kind} received ({n.get('doc_id')})"
        if t == ET.LCD_EVALUATED:
            miss = p.get("missing") or []
            ev = (p.get("checklist") or {}).get("evidence") or {}
            if not miss:
                first = next(iter(ev.values()), None)
                return f"LCD: all 5 criteria met{_q(first, 50)}"
            return f"LCD: missing {', '.join(miss)}"
        if t == ET.ADDENDUM_REQUESTED:
            return f"Addendum requested for {', '.join(p.get('missing') or [])}"
        if t == ET.SWO_SIGNED_RECEIVED:
            s = p.get("swo", {})
            return f"Signed SWO received: {s.get('hcpcs')} {s.get('item_description') or ''}".rstrip()
        if t == ET.SWO_EVALUATED:
            if p.get("complete"):
                return f"SWO complete, code {p.get('hcpcs')}"
            return f"SWO incomplete: {', '.join(p.get('missing') or [])} (code {p.get('hcpcs')})"
        if t == ET.CORRECTION_REQUESTED:
            return f"Correction requested: {', '.join(p.get('fields') or [])}"
        if t == ET.ORDER_VALIDATED:
            return "Order valid: K0001, SWO complete, LCD met"
        if t == ET.ORDER_SENT:
            return f"PCP faxed order to {_name(p.get('to_sid'))}"
        if t == ET.ORDER_RECEIVED:
            return f"{_name(p.get('by_sid'))} confirmed order received"
        if t == ET.ORDER_REDIRECT:
            return f"Order redirected {_short(p.get('from_sid'))} -> {_name(p.get('to_sid'))}"
        if t == ET.ORDER_STALLED:
            return f"Order stalled after {p.get('followups')} follow-ups"
        if t == ET.PCP_REFUSED:
            return f"PCP refused: {p.get('reason')}"
        if t == ET.CANDIDATES_RANKED:
            d = p.get("distances") or {}
            top = ", ".join(f"{_short(s)} {d.get(s, 0):.1f}mi" for s in (p.get("sids") or [])[:3])
            return f"{len(p.get('sids') or [])} suppliers ranked; nearest: {top}"
        if t == ET.SUPPLIER_NO_ANSWER:
            return f"{_name(p.get('sid'))}: no answer (attempt {p.get('attempt')})"
        if t == ET.SUPPLIER_QUALIFIED:
            f = p.get("facts") or {}
            ev = f.get("evidence") or {}
            q = (
                ev.get("k0001_in_stock")
                or ev.get("accepts_assignment_all_months")
                or next(iter(ev.values()), None)
            )
            days = f.get("earliest_delivery_days")
            extra = f", delivers in {days}d" if days is not None else ""
            return f"{_name(p.get('sid'))} qualified{extra}{_q(q, 55)}"
        if t == ET.SUPPLIER_DISQUALIFIED:
            return f"{_name(p.get('sid'))} disqualified: {p.get('reason')}"
        if t == ET.SUPPLIER_COMMITTED:
            until = _when(p.get("expires_at"))
            return f"Committed {_name(p.get('sid'))} (ref {p.get('ref')}, hold until {until})"
        if t == ET.SUPPLIER_REVERIFIED:
            return f"{_name(p.get('sid'))} re-verified stock"
        if t == ET.SUPPLIER_RELEASED:
            return f"Released {_name(p.get('sid'))}: {p.get('reason')}"
        if t == ET.SUPPLIERS_EXHAUSTED:
            return "All candidate suppliers exhausted"
        if t == ET.ASSIGNMENT_CONFIRMED:
            return f"{_name(p.get('sid'))} accepts assignment all rental months"
        if t == ET.ASSIGNMENT_REFUSED:
            return f"{_name(p.get('sid'))} refuses assignment"
        if t == ET.DELIVERY_SLOT_PROPOSED:
            return f"{_short(p.get('sid'))} proposes delivery {_when(p.get('slot'))}"
        if t == ET.DELIVERY_SCHEDULED:
            return f"Delivery scheduled with {_short(p.get('sid'))} for {_when(p.get('slot'))}"
        if t == ET.DELIVERY_MISSED:
            return "Delivery missed"
        if t == ET.DELIVERED:
            return f"Delivered by {_short(p.get('sid'))}"
        if t == ET.POD_CONFIRMED:
            return "Eleanor confirmed delivery (POD)"
        if t == ET.COMMITMENT_MADE:
            who = _name(p.get("party"))
            return (
                f"Ledger: {who} will {p.get('what')} by {_when(p.get('due_at'))}{_q(p.get('evidence'), 50)}"
            )
        if t == ET.COMMITMENT_KEPT:
            return f"Ledger: {p.get('cid')} kept"
        if t == ET.COMMITMENT_BROKEN:
            return f"Ledger: {p.get('cid')} BROKEN"
        if t == ET.CALL_COMPLETED:
            return f"Call {p.get('task_id')} -> {p.get('outcome')} [{p.get('transcript_id')}]"
        if t == ET.EXTRACTION_FAILED:
            return f"Extraction failed on {p.get('field')}: {p.get('reason')}"
        if t == ET.ESCALATION_OPENED:
            esc = p.get("escalation") or {}
            return f"ESCALATE {esc.get('trigger')} [{esc.get('eid')}]: {_clip(esc.get('question', ''), 80)}"
        if t == ET.HUMAN_RESOLVED:
            note = f" - {p.get('note')}" if p.get("note") else ""
            return f"{p.get('resolver')} resolved {p.get('eid')} -> {p.get('option_id')}{note}"
    except Exception:  # pragma: no cover - never crash rendering
        pass
    return _clip(str(p), 90)


def event_track(e: Event) -> str:
    if e.type == ET.ESCALATION_OPENED:
        return str((e.payload.get("escalation") or {}).get("track", "case"))
    if e.type == ET.HUMAN_RESOLVED:
        return "case"
    return TRACK_BY_TYPE.get(e.type, "case")


def event_row(event: Event) -> tuple[str, str, str, str, str]:
    """(sim time, track, event type, source chip label, summary)."""
    src = SOURCE_LABEL.get(event.source.kind, event.source.kind)
    return (sim_time_label(event.sim_time), event_track(event), event.type.value, src, event_summary(event))


def row_text(row: tuple[str, str, str, str, str], etype: EventType | None = None) -> Text:
    when, track, typ, src, summary = row
    t = Text()
    t.append(when.ljust(TIME_W), style="dim")
    t.append(track[: TRACK_W - 1].ljust(TRACK_W), style=TRACK_STYLE.get(track, ""))
    tstyle = TYPE_STYLE.get(etype, "") if etype else ""
    t.append(typ[: TYPE_W - 1].ljust(TYPE_W), style=tstyle or "bold")
    t.append(f" {src} ", style=SOURCE_STYLE.get(src, ""))
    t.append(" " * max(1, SRC_W - len(src) - 1))
    t.append(summary)
    return t


def _matches(e: Event, track: str | None, show_all: bool) -> bool:
    if track:
        tr = event_track(e)
        if track == "case":
            return tr == "case" or e.type in (ET.ESCALATION_OPENED, ET.HUMAN_RESOLVED)
        return tr == track or (track == "order" and tr == "ledger")
    return show_all or e.type not in QUIET_TYPES


def render_timeline(
    events: Iterable[Event],
    *,
    track: str | None = None,
    console: Console | None = None,
    show_all: bool = False,
) -> int:
    """Print the timeline with a header per sim day. Returns the number of rows printed."""
    con = console or Console()
    last_day = None
    n = 0
    for e in events:
        if not _matches(e, track, show_all):
            continue
        day = _ct(e.sim_time).date() if _ct(e.sim_time) else None
        if day != last_day:
            con.print(Rule(Text(day_header(e.sim_time), style="bold"), style="grey50", align="left"))
            last_day = day
        con.print(row_text(event_row(e), e.type), overflow="fold")
        n += 1
    if n == 0:
        con.print("[dim](no events)[/dim]")
    return n


def live_printer(
    console: Console | None = None, *, track: str | None = None, show_all: bool = False
) -> Callable[[Any], None]:
    """on_event callback: prints each event as it is appended, with day headers."""
    con = console or Console()
    state: dict[str, Any] = {"day": None}

    def on_event(e: Any, *_: Any, **__: Any) -> None:
        if not isinstance(e, Event) or not _matches(e, track, show_all):
            return
        d = _ct(e.sim_time)
        day = d.date() if d else None
        if day != state["day"]:
            con.print(Rule(Text(day_header(e.sim_time), style="bold"), style="grey50", align="left"))
            state["day"] = day
        con.print(row_text(event_row(e), e.type), overflow="fold")

    return on_event


# ------------------------------------------------------------------------------------------- panels

STATUS_STYLE = {
    "DONE": "bold green",
    "BLOCKED_ON_HUMAN": "bold yellow",
    "STUCK": "bold red",
    "MAX_STEPS": "bold red",
    "CANCELLED": "bold red",
}


def _days(a: Any, b: Any) -> str:
    da, db = _ct(a), _ct(b)
    if not (da and db):
        return "?"
    return f"{(db - da).total_seconds() / 86400:.1f}"


def render_summary(summary: Any, score: Any = None, *, console: Console | None = None) -> None:
    con = console or Console()
    s = _fields(summary)
    status = str(s.get("status", "?"))
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("case", Text(str(s.get("case_id")), style="bold"))
    t.add_row("scenario", str(s.get("scenario")))
    t.add_row("status", Text(status, style=STATUS_STYLE.get(status, "bold")))
    t.add_row(
        "sim time",
        f"{_when(s.get('sim_start'))} -> {_when(s.get('sim_end'))} "
        f"({_days(s.get('sim_start'), s.get('sim_end'))} days)",
    )
    t.add_row("steps / calls", f"{s.get('steps')} / {s.get('calls')}")
    esc = s.get("escalations") or []
    open_esc = s.get("open_escalations") or []
    t.add_row("escalations", ", ".join(esc) or "none")
    if open_esc:
        t.add_row("open", Text(", ".join(open_esc), style="bold yellow"))
    viol = s.get("invariant_violations") or []
    t.add_row(
        "invariants", Text("; ".join(viol), style="bold red") if viol else Text("all held", style="green")
    )
    cost = s.get("cost_usd")
    t.add_row("LLM usage", f"{s.get('usage_tokens') or 0} tokens" + (f", ${cost:.4f}" if cost else ""))
    if score is not None:
        sc = _fields(score)
        passed = sc.get("passed", sc.get("met_expectation"))
        if passed is not None:
            t.add_row(
                "expectation", Text("PASS", style="bold green") if passed else Text("FAIL", style="bold red")
            )
        for k in ("expected_status", "autonomous_close", "false_close", "phi_before_qualification"):
            if k in sc:
                t.add_row(k.replace("_", " "), str(sc[k]))
        reasons = sc.get("reasons") or sc.get("failures") or []
        if reasons:
            t.add_row("why", Text("; ".join(map(str, reasons)), style="red"))
    border = STATUS_STYLE.get(status, "white").replace("bold ", "")
    con.print(Panel(t, title="Run summary", border_style=border, expand=False))


def _fmt_rate(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def render_metrics(metrics: Any, scores: Iterable[Any], *, console: Console | None = None) -> None:
    con = console or Console()
    tbl = Table(title="Scenario results", header_style="bold")
    for col in (
        "scenario",
        "final",
        "expected",
        "pass",
        "sim days",
        "calls",
        "escalations",
        "false close",
        "inv. viol.",
    ):
        tbl.add_column(col)
    for sc in scores:
        d = _fields(sc)
        passed = d.get("passed", d.get("met_expectation"))
        final = str(d.get("status", d.get("final", "?")))
        viol = d.get("invariant_violations", 0)
        nviol = len(viol) if isinstance(viol, list | tuple) else viol
        esc = d.get("escalations_opened", d.get("escalations", []))
        tbl.add_row(
            str(d.get("name", d.get("scenario", "?"))),
            Text(final, style=STATUS_STYLE.get(final, "")),
            str(d.get("expected_status", d.get("expected_final", "?"))),
            Text("PASS", style="green") if passed else Text("FAIL", style="bold red"),
            _fmt_rate(d.get("sim_days")),
            str(d.get("calls", "?")),
            ", ".join(esc) if isinstance(esc, list | tuple) else str(esc),
            Text(str(d.get("false_close", False)), style="bold red" if d.get("false_close") else "dim"),
            Text(str(nviol), style="bold red" if nviol else "dim"),
        )
    con.print(tbl)
    m = _fields(metrics)
    g = Table.grid(padding=(0, 2))
    g.add_column(style="dim")
    g.add_column()
    g.add_row("scenarios", str(m.get("scenarios")))
    g.add_row("autonomous closes", str(m.get("autonomous_closes")))
    fc = m.get("false_closes", 0)
    g.add_row("false closes", Text(str(fc), style="bold red" if fc else "bold green"))
    g.add_row("escalation precision", _fmt_rate(m.get("escalation_precision")))
    g.add_row("escalation recall", _fmt_rate(m.get("escalation_recall")))
    iv = m.get("invariant_violations", 0)
    g.add_row("invariant violations", Text(str(iv), style="bold red" if iv else "green"))
    pl = m.get("phi_leaks", 0)
    g.add_row("PHI leaks", Text(str(pl), style="bold red" if pl else "green"))
    ok = m.get("all_passed")
    g.add_row("verdict", Text("ALL PASSED", style="bold green") if ok else Text("FAILURES", style="bold red"))
    con.print(Panel(g, title="Metrics", border_style="green" if ok else "red", expand=False))


def render_snapshot(snapshot: Any, *, console: Console | None = None) -> None:
    con = console or Console()
    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    tbl.add_column("track")
    tbl.add_column("status")
    tbl.add_column("who has it")
    tbl.add_column("when")
    for it in snapshot.items:
        tr = it.track.value if isinstance(it.track, Track) else str(it.track)
        tbl.add_row(
            Text(tr, style=TRACK_STYLE.get(tr, "")), it.plain_status, it.owner_label, it.due_label or ""
        )
    body = Table.grid()
    body.add_row(Text(snapshot.headline, style="bold"))
    body.add_row("")
    body.add_row(tbl)
    c = snapshot.cost
    body.add_row("")
    body.add_row(
        Text(
            f"Cost: ${c.deductible_usd} deductible, then {c.coinsurance_pct}% of the monthly rental "
            f"for {c.rental_months} months",
            style="dim",
        )
    )
    if snapshot.next_patient_action:
        body.add_row(Text(f"Needed from Eleanor: {snapshot.next_patient_action}", style="green"))
    con.print(
        Panel(body, title=f"Snapshot as of {_when(snapshot.as_of)}", border_style="green", expand=False)
    )


SPEAKER_STYLE = {"caller": "bold cyan", "counterparty": "bold yellow", "system": "dim"}


def render_transcript(record: Any, *, console: Console | None = None) -> None:
    """Accepts a store.TranscriptRecord or a bare CallResult."""
    con = console or Console()
    result = getattr(record, "result", record)
    head = Text()
    tid = getattr(record, "transcript_id", None)
    if tid:
        head.append(f"{tid}  ", style="bold")
    party = getattr(record, "party_id", None)
    if party:
        head.append(f"{_name(party)}  ", style="cyan")
    if getattr(record, "sim_time", None):
        head.append(f"{_when(record.sim_time)}  ", style="dim")
    oc = result.outcome
    head.append(f"{oc}", style="bold green" if oc == "COMPLETED" else "bold yellow")
    head.append(f"  transport={result.transport}  {result.duration_s}s", style="dim")
    con.print(head)
    for u in result.transcript:
        line = Text()
        line.append(f"{u.turn:>2} ", style="dim")
        line.append(f"{u.name}: ", style=SPEAKER_STYLE.get(u.speaker, ""))
        line.append(u.text)
        if u.event and u.event != "none":
            line.append(f"  [{u.event}]", style="magenta")
        if u.heard_text and u.heard_text != u.text:
            line.append(f"  (heard: {u.heard_text})", style="dim")
        con.print(line, overflow="fold")
    if result.claims:
        ct = Table(title="Extracted claims", header_style="bold", title_justify="left")
        ct.add_column("field")
        ct.add_column("status")
        ct.add_column("value")
        ct.add_column("evidence quote", overflow="fold")
        for c in result.claims:
            ct.add_row(
                c.field,
                Text(c.status, style="green" if c.status == "stated" else "yellow"),
                c.value or "",
                c.evidence_quote or "",
            )
        con.print(ct)
    if result.flags:
        con.print(Text("flags: " + ", ".join(result.flags), style="yellow"))


def render_transcript_list(records: Iterable[Any], *, console: Console | None = None) -> None:
    con = console or Console()
    tbl = Table(header_style="bold")
    for col in ("transcript", "sim time", "party", "task", "outcome", "transport", "turns", "stated claims"):
        tbl.add_column(col, overflow="fold")
    for r in records:
        res = r.result
        stated = [c.field for c in res.claims if c.status == "stated"]
        tbl.add_row(
            r.transcript_id,
            sim_time_label(r.sim_time),
            _short(r.party_id),
            r.task_id.split(":", 1)[-1],
            res.outcome,
            res.transport,
            str(len(res.transcript)),
            ", ".join(stated),
        )
    con.print(tbl)


def render_escalations(escalations: Iterable[Any], *, console: Console | None = None) -> None:
    """Accepts models.Escalation objects or dicts (e.g. store.cached_escalations rows / model dumps)."""
    con = console or Console()
    items = list(escalations)
    if not items:
        con.print("[green]No escalations.[/green]")
        return
    for esc in items:
        d = _fields(esc)
        if "body_json" in d and "options" not in d:  # cached row
            import json

            d = {**json.loads(d["body_json"]), **{k: v for k, v in d.items() if k != "body_json"}}
        status = str(d.get("status"))
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim")
        g.add_column()
        g.add_row("lane / track", f"{d.get('lane')} / {d.get('track')}")
        g.add_row("question", Text(str(d.get("question", "")), style="bold"))
        sugg = d.get("suggested")
        for o in d.get("options") or []:
            od = _fields(o)
            mark = " (suggested)" if od.get("option_id") == sugg else ""
            g.add_row(
                "option",
                Text(f"{od.get('option_id')}: {od.get('label')}{mark}", style="green" if mark else ""),
            )
        ctx = d.get("context") or {}
        for k, v in ctx.items():
            if k in ("key",):
                continue
            g.add_row(f"ctx.{k}", _clip(str(v), 100))
        if d.get("resolution"):
            g.add_row("resolution", str(d.get("resolution")))
        title = f"{d.get('eid')}  {d.get('trigger')}  [{status}]"
        con.print(
            Panel(
                g,
                title=title,
                title_align="left",
                border_style="yellow" if status == "OPEN" else "dim",
                expand=False,
            )
        )
