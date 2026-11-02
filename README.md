# Mira Mace — DME coordination prototype

A working slice of the system that runs Eleanor Martinez's Medicare wheelchair case (standard manual chair, K0001) with the care advocate asleep. It calls Dr. Chen's office for the written order, screens and commits a supplier, reads the visit note against the Medicare coverage criteria, keeps Eleanor informed through a companion agent, and recovers from the failure modes in the brief: supplier can't serve, supplier ghosts, request lost, wrong code, cost worry. When it hits something it should not decide alone, it opens a typed escalation and blocks only that track. Every run is an append-only event log you can replay.

**Principle: AI owns the conversations. Code owns the commitments.** LLMs talk (to Eleanor, to the PCP front desk, to suppliers) and turn speech and documents into claims, each backed by a verbatim quote. Deterministic code decides every state change, owner, due date, retry, failover, escalation and every fact Eleanor is told.

Writeup: [`docs/WRITEUP.md`](docs/WRITEUP.md) · Live-defense run sheet: [`docs/DEMO.md`](docs/DEMO.md) · Design: [`docs/architecture.md`](docs/architecture.md), [`docs/user-journey.md`](docs/user-journey.md)

---

## Quickstart (offline, no API key)

```bash
uv sync                               # Python 3.12
uv run mira run-all --llm fake        # all 7 scenarios + metrics, ~seconds, no network
uv run mira run happy_path            # stream one case's timeline on the virtual clock
uv run pytest -q                      # 356 passed, 1 skipped
```

`--llm fake` swaps every LLM for a deterministic fake behind the same ports (fact-sheet personas, rule-based caller, keyword companion). The core, rules, guards and checks are the same code in both modes.

## Live run (real LLM calls)

```bash
cp .env.example .env
# OPENAI_API_KEY=sk-...
# MIRA_TEXT_MODEL=<model id>          # caller, persona, extractor, LCD reader, companion
# MIRA_TURN_MODEL=<faster model id>   # optional: per-turn call brains (latency)
uv run mira run happy_path --llm live
uv run mira chat <case_id> --llm live # talk to the companion as Eleanor
```

The run summary prints tokens and USD per case (from the LLM ledger, when the model has a known list price).

## Voice (PCP leg over LiveKit)

```bash
brew install livekit livekit-cli
livekit-server --dev                  # terminal A: ws://localhost:7880, devkey/secret
uv sync --extra voice                 # terminal B
uv run mira call pcp --voice --listen # opens a listener link: hear the AI caller and "Maria"
```

Two in-process LiveKit agent sessions (caller ↔ front-desk persona), OpenAI STT/TTS, VAD turn detection. Any voice failure falls back to text automatically (`transport=text_fallback`, flag `voice_failed:<reason>`). Setup and troubleshooting: [`docs/voice-smoke.md`](docs/voice-smoke.md).

## Commands

| Command | What it does |
|---|---|
| `mira run <scenario> [--llm fake\|live] [--voice-pcp] [--quiet] [--all]` | Run one scenario end to end on the virtual clock; stream the timeline; print summary + `case_id` |
| `mira run-all [--llm fake\|live] [-v]` | Run every scenario, print results + metrics; exit 1 on any false close, invariant violation or failed expectation |
| `mira timeline <case_id> [--track order\|supplier\|patient\|delivery\|assignment\|case] [--all]` | Replay a case's event log |
| `mira snapshot <case_id>` | The patient-facing snapshot — the companion's only fact source |
| `mira transcripts <case_id>` | List call transcripts (party, outcome, transport, stated claims) |
| `mira transcript <tr_id> --case <case_id>` | One transcript with extracted claims and evidence quotes |
| `mira escalations <case_id> [--open]` | Escalations: question, typed options, suggested option, context |
| `mira resolve <case_id> <eid> <option> [--note ...] [--by ...]` | Answer as the human; emits `HUMAN_RESOLVED` and resumes the case |
| `mira chat <case_id> [--llm fake\|live] [--resume]` | REPL as Eleanor; `/snapshot` shows what the companion knows |
| `mira call pcp [--text\|--voice] [--listen] [--llm live\|fake]` | Standalone PCP call demo (text or LiveKit voice) |
| `mira version` | Package version |

Runs are stored in `runs/<case_id>.db`; every command after `run` reads that file.

## What is mocked vs real

| Piece | In this prototype |
|---|---|
| Telephony | **Mocked.** Calls run as text turn-by-turn between a caller agent and a counterparty agent. The PCP leg can run as real audio over local LiveKit (`mira call pcp --voice`). No PSTN. |
| Counterparties (Maria at Sunrise, 12 suppliers) | LLM persona with a **hidden fact sheet** per scenario; reveals a fact only when asked; behaviours: helpful, partial, IVR, transfer, callback-later, never-got-it, injection |
| Inbound faxes (visit note, addendum, signed SWO, proof of delivery) | Simulator driven by the scenario's `sim` block (delays in business hours) |
| Patient (Eleanor) | Scripted replies in `run` (per scenario); **live** in `mira chat` |
| LLM calls | **Real** in `--llm live` (OpenAI Responses API, structured outputs): caller turns, persona turns, two-pass extraction, LCD reading, companion. Deterministic fakes in `--llm fake` |
| Supplier directory | The 12 suppliers from the brief (`data/suppliers.csv`); distance = ZIP-centroid heuristic, no geocoding |
| Database | Local SQLite per run (`runs/<case_id>.db`): append-only event log + task outbox; state is a pure projection |
| Clock | Virtual clock (business hours 9–17 CT, patient hours); sim starts Mon 2026-11-02 10:00 CT |
| Claims submission, PCP portal, auth, HIPAA controls | Not built (designed for; see writeup) |

## Scenarios

| Scenario | What it proves | Expected |
|---|---|---|
| `happy_path` | Full loop: screen → commit → order requested → LCD met → SWO valid → order faxed to the committed supplier → re-verify → slot → delivered → POD | `DONE`, 0 escalations |
| `supplier_disqualified` | Eligibility is a rule over quoted facts: Lakeshore out of K0001, Lincoln Park won't take assignment every month → Halsted qualifies | `DONE` |
| `ghost_failover` | Yes-then-silent: committed supplier stops answering after the order is faxed → lease released, backup committed, `ORDER_REDIRECT` to the PCP | `DONE` |
| `pcp_never_got_it` | Request falls in a hole: missed promise → verify first ("did you receive request #R?") → resend as v2 under a new idempotency key | `DONE` |
| `lcd_missing_addendum` | Visit note lacks "home accommodates the chair" → addendum request names that criterion → re-evaluated | `DONE` |
| `cost_concern` | "I can't afford that" → patient-lane escalation to a nurse; only the patient track waits; `mira resolve` resumes | `BLOCKED_ON_HUMAN` (→ `DONE` after resolve) |
| `code_drift_k0003` | Signed order comes back K0003 → coding drift is a hard stop for a human | `BLOCKED_ON_HUMAN` (`CODE_DRIFT`) |

Schema and fact-sheet slots: [`scenarios/README.md`](scenarios/README.md).

## Results (from `mira run-all --llm fake` and `pytest`)

| Scenario | Final | Sim days | Calls |
|---|---|---|---|
| happy_path | DONE | 4.0 | 8 |
| supplier_disqualified | DONE | 4.0 | 10 |
| ghost_failover | DONE | 4.0 | 12 |
| lcd_missing_addendum | DONE | 8.0 | 11 |
| pcp_never_got_it | DONE | 10.0 | 12 |
| cost_concern | BLOCKED_ON_HUMAN | 3.2 | 7 |
| code_drift_k0003 | BLOCKED_ON_HUMAN | 3.0 | 5 |

7/7 expectations pass · 5 autonomous closes · **false closes 0** · escalation precision 1.00 / recall 1.00 · invariant violations 0 · PHI before qualification 0 · `pytest`: 356 passed, 1 skipped (the skip needs the voice extra).

These numbers are from fake mode: they prove the orchestration, rules and guards. Extraction accuracy, companion tone and per-case cost need the live runs (see the writeup's "What's next").

## Repo map

```
src/miramace/
  models.py events.py ports.py clock.py     contracts: entities, 51 event types, async ports, business-hours clock
  store.py reducer.py invariants.py         SQLite event log + task outbox, pure reducer -> CaseState, 5 invariants
  rules.py geo.py planner.py escalation.py  eligibility, SWO/LCD/K0001 rules, ranking, pure planner, escalation catalogue
  executor.py interpret.py runloop.py       PHI gate + CallTask build, claims -> events (quote re-check), run loop
  snapshot.py notify.py grounding.py intents.py   patient read model, when-to-notify policy, grounding check, Intent API
  calls/       runner.py (phases, disclosure, guards, readback, finalize) · brains.py (LLM caller/persona)
  extract/     extractor.py (two-pass, quote verification) · lcd.py (visit note -> LCD checklist)
  companion/   agent.py (LLM companion + grounding regen + distress safety net)
  agents/      prompts.py · schemas.py (strict structured-output models)
  llm/         port.py (OpenAI Responses, ledger, cost)
  voice/       transport.py (LiveKit two-agent call, listener link, text fallback)
  fakes/       fact-sheet personas, doc reader, inbound simulator, scripted patient, keyword companion
  scenario.py metrics.py render.py cli.py
scenarios/     7 YAML scenarios with hidden fact sheets + expectations
data/          suppliers.csv (the brief's directory)
docs/          WRITEUP.md DEMO.md architecture.md user-journey.md voice-smoke.md problem.md ...
tests/         unit + scenario tests (no network)
```
