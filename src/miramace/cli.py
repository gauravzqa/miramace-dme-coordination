"""B7 — `mira` console entry point.

Commands: run, run-all, timeline, snapshot, transcripts, transcript, escalations, resolve, chat,
call pcp, version.
Engine modules (scenario/runloop/metrics) are imported lazily inside commands so the CLI loads even while they
are being built. Live mode requires OPENAI_API_KEY and MIRA_TEXT_MODEL (loaded from .env).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miramace import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Mira Mace DME coordination prototype.")
call_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Standalone call demos.")
app.add_typer(call_app, name="call")

console = Console(highlight=False)

TRACKS = ("order", "supplier", "patient", "delivery", "assignment", "case")


REPO_ROOT = Path(__file__).resolve().parents[2]  # src/miramace/cli.py -> repo root (editable install)


def env_files() -> list[Path]:
    """Candidate .env files, highest priority first: CWD, then the repo root (package parent)."""
    out: list[Path] = []
    for p in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if p.is_file() and p.resolve() not in {q.resolve() for q in out}:
            out.append(p)
    return out


def load_env() -> list[Path]:
    """Load .env from CWD and the repo root without overriding real environment variables (the first file
    to set a key wins). Returns the files found."""
    files = env_files()
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - dotenv optional at runtime
        return files
    for f in files:
        try:
            load_dotenv(f, override=False)
        except Exception:  # pragma: no cover
            pass
    return files


@app.callback()
def main() -> None:
    """Mira Mace CLI."""
    load_env()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"miramace {__version__}")


# ------------------------------------------------------------------------------------------- helpers


def _fail(msg: str, code: int = 2) -> None:
    console.print(f"[bold red]error:[/bold red] {msg}")
    raise typer.Exit(code)


def _check_mode(llm: str) -> str:
    mode = llm.strip().lower()
    if mode not in ("fake", "live"):
        _fail(f"--llm must be 'fake' or 'live', got {llm!r}")
    if mode == "live":
        missing = [k for k in ("OPENAI_API_KEY", "MIRA_TEXT_MODEL") if not os.environ.get(k, "").strip()]
        if missing:
            looked = ", ".join(str(p) for p in (Path.cwd() / ".env", REPO_ROOT / ".env"))
            hint = {
                "OPENAI_API_KEY": "OPENAI_API_KEY=sk-...",
                "MIRA_TEXT_MODEL": "MIRA_TEXT_MODEL=<model id, e.g. gpt-4o-mini>",
            }
            _fail(
                f"--llm live needs {', '.join(missing)}. Add "
                + "; ".join(hint[k] for k in missing)
                + f" to .env (looked in: {looked}) or export it; see .env.example"
            )
    return mode


def _set_llm_budget(max_llm_calls: int | None) -> None:
    """--max-llm-calls -> MIRA_MAX_LLM_CALLS (read by OpenAILLM; default 400 API requests per run)."""
    if max_llm_calls is not None:
        if max_llm_calls < 1:
            _fail("--max-llm-calls must be >= 1")
        os.environ["MIRA_MAX_LLM_CALLS"] = str(max_llm_calls)


def _engine(name: str) -> Any:
    import importlib

    try:
        return importlib.import_module(f"miramace.{name}")
    except ImportError as exc:  # pragma: no cover
        _fail(f"engine module miramace.{name} unavailable: {exc}")


def _need(mod: Any, attr: str) -> Any:
    fn = getattr(mod, attr, None)
    if fn is None:
        _fail(f"{mod.__name__}.{attr} is not implemented yet")
    return fn


def _open(case_id: str, db_dir: str) -> tuple[Any, Any]:
    """(Store, VirtualClock) for an existing case — via scenario.open_case, else a direct fallback."""
    scen = _engine("scenario")
    opener = getattr(scen, "open_case", None)
    if opener is not None:
        try:
            return opener(case_id, db_dir=db_dir)
        except FileNotFoundError as exc:
            _fail(str(exc))
    from miramace.clock import VirtualClock
    from miramace.store import Store

    path = Path(db_dir) / f"{case_id}.db"
    if not path.exists():
        _fail(f"no database for case {case_id!r} at {path}")
    store = Store(path)
    evs = store.events(case_id)
    if not evs:
        _fail(f"case {case_id!r} has no events in {path}")
    return store, VirtualClock(evs[-1].sim_time)


def _scenario_of(store: Any, case_id: str) -> str:
    row = store.case(case_id) or {}
    name = row.get("scenario")
    if not name:
        _fail(f"case {case_id!r} has no scenario recorded")
    return name


def _score(summary: Any, scenario: Any, store: Any) -> Any:
    try:
        metrics = _engine("metrics")
        return metrics.score(summary, scenario, store.state(summary.case_id), store)
    except Exception as exc:
        console.print(f"[yellow]scoring unavailable: {exc}[/yellow]")
        return None


# ----------------------------------------------------------------------------------------------- run


@app.command()
def run(
    scenario: str = typer.Argument(..., help="Scenario name (scenarios/<name>.yaml) or path."),
    llm: str = typer.Option("fake", "--llm", help="fake | live"),
    voice_pcp: bool = typer.Option(
        False, "--voice-pcp", help="Use the LiveKit voice transport for PCP calls."
    ),
    db_dir: str = typer.Option("runs", "--db-dir", help="Directory for runs/<case_id>.db"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Do not stream the timeline."),
    show_all: bool = typer.Option(
        False, "--all", help="Include audit events (CALL_COMPLETED) in the timeline."
    ),
    max_llm_calls: int = typer.Option(
        None, "--max-llm-calls", help="Live only: hard cap on OpenAI requests (env MIRA_MAX_LLM_CALLS, 400)."
    ),
) -> None:
    """Run one scenario end to end on the virtual clock, streaming the timeline."""
    from miramace import render

    mode = _check_mode(llm)
    _set_llm_budget(max_llm_calls)
    scen = _engine("scenario")
    run_scenario = _need(scen, "run_scenario")
    on_event = None if quiet else render.live_printer(console, show_all=show_all)
    if not quiet:
        console.rule(f"[bold]mira run {scenario}[/bold]  llm={mode}{'  voice-pcp' if voice_pcp else ''}")
    summary, store, scenario_obj = asyncio.run(
        run_scenario(scenario, mode=mode, voice_pcp=voice_pcp, db_dir=db_dir, on_event=on_event)
    )
    score = _score(summary, scenario_obj, store)
    render.render_summary(summary, score, console=console)
    if summary.open_escalations:
        console.print(f"[yellow]Open escalations:[/yellow] mira escalations {summary.case_id} --open")
    console.print(f"case_id: [bold]{summary.case_id}[/bold]")


@app.command("run-all")
def run_all(
    llm: str = typer.Option("fake", "--llm", help="fake | live"),
    db_dir: str = typer.Option("runs", "--db-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream every scenario's timeline."),
    max_llm_calls: int = typer.Option(
        None,
        "--max-llm-calls",
        help="Live only: cap on OpenAI requests PER scenario (env MIRA_MAX_LLM_CALLS).",
    ),
) -> None:
    """Run every scenario and print the metrics table. Exit 1 on any false close, invariant violation or
    failed expectation."""
    from miramace import render

    mode = _check_mode(llm)
    _set_llm_budget(max_llm_calls)
    scen = _engine("scenario")
    metrics = _engine("metrics")
    names = _need(scen, "list_scenarios")()
    scores = []
    failed_runs: list[str] = []
    for name in names:
        console.rule(f"[bold]{name}[/bold]")
        on_event = render.live_printer(console) if verbose else None
        try:
            summary, store, scenario_obj = asyncio.run(
                scen.run_scenario(name, mode=mode, db_dir=db_dir, on_event=on_event)
            )
        except Exception as exc:
            console.print(f"[bold red]{name} crashed:[/bold red] {type(exc).__name__}: {exc}")
            failed_runs.append(name)
            continue
        sc = metrics.score(summary, scenario_obj, store.state(summary.case_id), store)
        scores.append(sc)
        status = str(summary.status)
        style = render.STATUS_STYLE.get(status, "")
        console.print(
            f"  [{style}]{status}[/{style}]  case {summary.case_id}  "
            f"escalations={','.join(summary.escalations) or '-'}"
        )
    agg = metrics.aggregate(scores)
    render.render_metrics(agg, scores, console=console)
    bad = (
        bool(failed_runs)
        or (getattr(agg, "false_closes", 0) or 0) > 0
        or (getattr(agg, "invariant_violations", 0) or 0) > 0
        or not getattr(agg, "all_passed", False)
    )
    if failed_runs:
        console.print(f"[bold red]crashed: {', '.join(failed_runs)}[/bold red]")
    if bad:
        raise typer.Exit(1)


# ------------------------------------------------------------------------------------------ inspect


@app.command()
def timeline(
    case_id: str,
    track: str = typer.Option(None, "--track", help="order|supplier|patient|delivery|assignment|case"),
    db_dir: str = typer.Option("runs", "--db-dir"),
    show_all: bool = typer.Option(False, "--all", help="Include audit events."),
) -> None:
    """Print a case's event timeline (optionally one track)."""
    from miramace import render

    if track and track not in TRACKS:
        _fail(f"--track must be one of {', '.join(TRACKS)}")
    store, _ = _open(case_id, db_dir)
    render.render_timeline(store.events(case_id), track=track, console=console, show_all=show_all)


@app.command()
def snapshot(case_id: str, db_dir: str = typer.Option("runs", "--db-dir")) -> None:
    """Print the patient-facing snapshot (the companion's only fact source)."""
    from miramace import render
    from miramace.snapshot import build_snapshot

    store, clock = _open(case_id, db_dir)
    render.render_snapshot(build_snapshot(store.state(case_id), clock.now()), console=console)


@app.command()
def transcripts(case_id: str, db_dir: str = typer.Option("runs", "--db-dir")) -> None:
    """List call transcripts for a case."""
    from miramace import render

    store, _ = _open(case_id, db_dir)
    intents = {t.task_id: t.intent for t in store.tasks(case_id)}
    render.render_transcript_list(store.transcripts(case_id), console=console, intents=intents)


@app.command()
def transcript(
    transcript_id: str,
    case: str = typer.Option(..., "--case", help="Case id the transcript belongs to."),
    db_dir: str = typer.Option("runs", "--db-dir"),
) -> None:
    """Show one call transcript with extracted claims and evidence quotes."""
    from miramace import render

    store, _ = _open(case, db_dir)
    rec = store.transcript(transcript_id)
    if rec is None:
        _fail(f"no transcript {transcript_id!r} in case {case!r}")
    render.render_transcript(rec, console=console)


@app.command()
def escalations(
    case_id: str,
    open_only: bool = typer.Option(False, "--open", help="Only open escalations."),
    db_dir: str = typer.Option("runs", "--db-dir"),
) -> None:
    """Show escalations (question, typed options, suggestion)."""
    from miramace import render

    store, _ = _open(case_id, db_dir)
    items = list(store.state(case_id).escalations.values())
    if open_only:
        items = [e for e in items if str(e.status) == "OPEN"]
    render.render_escalations(items, console=console)
    for e in items:
        if str(e.status) == "OPEN":
            console.print(
                f'[dim]resolve:[/dim] mira resolve {case_id} {e.eid} {e.suggested or "<option>"} --note "..."'
            )


@app.command()
def resolve(
    case_id: str,
    eid: str,
    option: str,
    note: str = typer.Option(None, "--note"),
    by: str = typer.Option("advocate", "--by"),
    llm: str = typer.Option("fake", "--llm", help="fake | live"),
    db_dir: str = typer.Option("runs", "--db-dir"),
) -> None:
    """Resolve an escalation as a human and resume the case."""
    from miramace import render

    mode = _check_mode(llm)
    scen = _engine("scenario")
    runloop = _engine("runloop")
    store, clock = _open(case_id, db_dir)
    state = store.state(case_id)
    esc = state.escalations.get(eid)
    if esc is None:
        _fail(
            f"unknown escalation {eid!r}; open ones: "
            f"{', '.join(e.eid for e in state.escalations.values() if str(e.status) == 'OPEN') or 'none'}"
        )
    valid = [o.option_id for o in esc.options]
    if option not in valid:
        _fail(f"option {option!r} not valid for {esc.trigger}; choose from {', '.join(valid)}")
    scenario_obj = _need(scen, "load_scenario")(_scenario_of(store, case_id))
    ports = _need(scen, "build_ports")(scenario_obj, store, clock, mode=mode)
    before = len(store.events(case_id))
    summary = asyncio.run(
        _need(runloop, "resume_after_resolution")(case_id, eid, option, note, by, ports, store, clock)
    )
    new = store.events(case_id)[before:]
    console.rule(f"[bold]resolved {eid} -> {option}[/bold]")
    render.render_timeline(new, console=console)
    render.render_summary(summary, _score(summary, scenario_obj, store), console=console, resumed=True)


# ---------------------------------------------------------------------------------------------- chat


def _last_purpose(store: Any, case_id: str) -> str | None:
    from miramace.events import EventType

    for e in reversed(store.events(case_id)):
        if e.type == EventType.PATIENT_NOTIFIED:
            return e.payload.get("purpose")
    return None


_CONCERN_INTENTS = ("cost_concern", "needs_human")


def _submit_intents(api: Any, case_id: str, intents: list[Any], store: Any) -> list[tuple[Any, Any]]:
    """Submit each intent; a concern rejected because the case is closed is re-routed as needs_human
    (allowed on a closed case) so the care team still hears about it."""
    out: list[tuple[Any, Any]] = []
    for intent in intents:
        ack = api.submit(case_id, intent)
        out.append((intent, ack))
        if not ack.accepted and intent.kind in _CONCERN_INTENTS and intent.kind != "needs_human":
            fb = intent.model_copy(update={"kind": "needs_human", "note": "human"})
            out.append((fb, api.submit(case_id, fb)))
    return out


def _honest_reply(reply: str, acks: list[tuple[Any, Any]], state: Any) -> str:
    """Keep the companion's reply unless an intent it relied on was rejected; else say what happened."""
    rejected = [(i, a) for i, a in acks if not a.accepted]
    if not rejected:
        return reply
    routed = any(a.accepted and i.kind in _CONCERN_INTENTS for i, a in acks)
    closed = str(getattr(state.case, "state", "")) in (
        "DONE",
        "CANCELLED",
        "CaseStatus.DONE",
        "CaseStatus.CANCELLED",
    )
    if closed:
        if routed:
            return (
                "Your case is closed — I've let our care team know you reached out; "
                "a nurse can call you back."
            )
        return (
            "Your case is closed, so I couldn't record that here. "
            "Please call our care team and a nurse can help."
        )
    if routed:
        return "I've let our care team know you reached out; a nurse can call you back."
    reason = rejected[0][1].reason or "it doesn't match where your order is right now"
    return f"I wasn't able to record that just now ({reason}). Nothing has changed on your order."


async def _chat_loop(case_id: str, mode: str, db_dir: str, resume: bool) -> None:
    from miramace.intents import IntentAPI
    from miramace.models import Utterance
    from miramace.snapshot import build_snapshot

    store, clock = _open(case_id, db_dir)
    conv = f"chat-{case_id}"
    if mode == "live":
        from miramace.companion.agent import LLMCompanion
        from miramace.llm.port import make_llm

        st = store.state(case_id)
        profile = st.patient_info.profile if st.patient_info else None
        companion: Any = LLMCompanion(make_llm("live"), conversation_id=conv, profile=profile)
    else:
        from miramace.fakes.companion import KeywordCompanion

        companion = KeywordCompanion(conversation_id=conv)
        companion.last_purpose = _last_purpose(store, case_id)
    api = IntentAPI(store, clock)
    history: list[Utterance] = []
    console.print(
        f"[dim]Chatting as Eleanor on case {case_id} (llm={mode}). /quit to exit, "
        f"/snapshot to see what the companion knows.[/dim]"
    )
    while True:
        try:
            text = (await asyncio.to_thread(console.input, "[bold green]Eleanor>[/bold green] ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/snapshot":
            from miramace import render

            render.render_snapshot(build_snapshot(store.state(case_id), clock.now()), console=console)
            continue
        snap = build_snapshot(store.state(case_id), clock.now())
        reply = await companion.respond(snap, history, text)
        # Submit intents FIRST: the reply must never promise something the case then rejects.
        acks = _submit_intents(api, case_id, reply.intents, store)
        say = _honest_reply(reply.reply, acks, store.state(case_id))
        ts = float(len(history))
        history.append(Utterance(turn=len(history), speaker="counterparty", name="Eleanor", text=text, ts=ts))
        history.append(Utterance(turn=len(history), speaker="caller", name="Mira", text=say, ts=ts + 1))
        console.print(f"[bold cyan]Mira>[/bold cyan] {say}")
        if reply.flags or not reply.grounding_ok:
            flags = list(reply.flags) + ([] if reply.grounding_ok else ["grounding_failed"])
            console.print(f"[yellow]  flags: {', '.join(flags)}[/yellow]")
        unblocked = False
        for intent, ack in acks:
            style = "green" if ack.accepted else "red"
            detail = ", ".join(ack.events) or "-"
            reason = f" ({ack.reason})" if ack.reason else ""
            console.print(
                f"[{style}]  intent {intent.kind}: {'accepted' if ack.accepted else 'rejected'} "
                f"-> {detail}{reason}[/{style}]"
            )
            if ack.accepted and intent.kind not in ("ask_status",):
                unblocked = True
        if resume and unblocked:
            await _resume_case(case_id, store, clock, mode, db_dir)


async def _resume_case(case_id: str, store: Any, clock: Any, mode: str, db_dir: str) -> None:
    from miramace import render

    scen = _engine("scenario")
    runloop = _engine("runloop")
    run_case = getattr(runloop, "run_case", None)
    if run_case is None:
        console.print("[yellow]  (runloop.run_case unavailable; resume skipped)[/yellow]")
        return
    scenario_obj = scen.load_scenario(_scenario_of(store, case_id))
    ports = scen.build_ports(scenario_obj, store, clock, mode=mode)
    before = len(store.events(case_id))
    try:
        summary = await run_case(case_id, ports, store, clock)
    except TypeError:
        summary = await run_case(case_id, ports=ports, store=store, clock=clock)
    render.render_timeline(store.events(case_id)[before:], console=console)
    console.print(f"[dim]  case status: {getattr(summary, 'status', '?')}[/dim]")


@app.command()
def chat(
    case_id: str,
    llm: str = typer.Option("fake", "--llm", help="fake | live"),
    resume: bool = typer.Option(False, "--resume", help="Run the case forward after an accepted intent."),
    db_dir: str = typer.Option("runs", "--db-dir"),
) -> None:
    """Talk to the companion as Eleanor (REPL)."""
    mode = _check_mode(llm)
    asyncio.run(_chat_loop(case_id, mode, db_dir, resume))


# ------------------------------------------------------------------------------------------ call pcp


def demo_pcp_task() -> Any:
    from miramace.models import CallTask, SlotSpec, eleanor

    slots = [
        SlotSpec(
            id="request_received",
            ask="Did you receive our order request for her wheelchair?",
            type="bool",
            required=True,
        ),
        SlotSpec(
            id="order_status", ask="What is the status of the wheelchair order?", type="str", required=True
        ),
        SlotSpec(id="promised_action", ask="What will the office send us?", type="str", required=True),
        SlotSpec(id="promised_by", ask="When can we expect it?", type="str", required=True),
        SlotSpec(id="named_owner", ask="Who is handling it?", type="str", required=True),
        SlotSpec(id="fax_number", ask="What fax number should we use?", type="str", required=False),
    ]
    return CallTask(
        task_id="demo:pcp:request_order:v1",
        party="pcp",
        party_id="pcp",
        org_name="Sunrise Family Medicine",
        phone="(312) 555-0198",
        intent="request_order",
        purpose="Request the visit note and a signed standard written order for a K0001 manual wheelchair.",
        slots=slots,
        phi=eleanor().phi(),
        context={"visit": "a patient of Dr. Chen seen 3 days ago"},
    )


def demo_maria() -> Any:
    from miramace.models import Fact, PersonaSpec

    facts = [
        Fact(
            id="f_recv",
            slot="request_received",
            value="yes",
            text="Yes, we got the request yesterday.",
            reveal_when="asked whether the request was received",
        ),
        Fact(
            id="f_status",
            slot="order_status",
            value="pending signature",
            text="It's waiting on Dr. Chen's signature.",
            reveal_when="asked about order status",
        ),
        Fact(
            id="f_action",
            slot="promised_action",
            value="fax visit note and signed order",
            text="We'll fax the visit note and the signed order.",
            reveal_when="asked what will be sent",
        ),
        Fact(id="f_by", slot="promised_by", value="Thursday", text="By Thursday.", reveal_when="asked when"),
        Fact(
            id="f_owner",
            slot="named_owner",
            value="Maria",
            text="I'm handling it myself, this is Maria.",
            reveal_when="asked who owns it",
        ),
        Fact(
            id="f_fax",
            slot="fax_number",
            value="312-555-0199",
            text="Our fax is 312-555-0199.",
            reveal_when="asked for the fax number",
        ),
    ]
    return PersonaSpec(
        party_id="pcp",
        role="pcp",
        org_name="Sunrise Family Medicine",
        answerer_name="Maria",
        behaviour="helpful",
        facts=facts,
    )


async def text_pcp_call(mode: str) -> Any:
    from miramace.calls.runner import TextCallerPort
    from miramace.clock import VirtualClock

    spec = demo_maria()
    if mode == "live":
        from miramace.calls.brains import LLMCallerBrain, LLMPersonaBrain
        from miramace.extract.extractor import Extractor
        from miramace.llm.port import make_llm

        llm = make_llm("live")
        port = TextCallerPort(
            LLMCallerBrain(llm), LLMPersonaBrain(llm, spec), Extractor(llm), clock=VirtualClock()
        )
    else:
        from miramace.calls.brains import FakeCallerBrain, FakePersonaBrain
        from miramace.extract.extractor import FakeExtractor

        port = TextCallerPort(
            FakeCallerBrain(), FakePersonaBrain(spec), FakeExtractor({"pcp": spec}), clock=VirtualClock()
        )
    return await port.run_call(demo_pcp_task())


@call_app.command("pcp")
def call_pcp(
    voice: bool = typer.Option(False, "--voice/--text", help="LiveKit voice (Mac) or text transport."),
    listen: bool = typer.Option(False, "--listen", help="Print a listener link to hear the voice call."),
    llm: str = typer.Option("live", "--llm", help="live | fake (text only; voice is always live)"),
) -> None:
    """Standalone PCP call demo (caller agent <-> Maria at Sunrise Family Medicine)."""
    from miramace import render

    if voice:
        _check_mode("live")
        try:
            from miramace.voice.transport import smoke
        except ImportError as exc:
            _fail(f"voice transport unavailable ({exc}); install with: uv sync --extra voice")
        result = asyncio.run(smoke(listen))
    else:
        mode = _check_mode(llm)
        result = asyncio.run(text_pcp_call(mode))
    if result is not None and hasattr(result, "transcript"):
        render.render_transcript(result, console=console)


if __name__ == "__main__":
    app()
