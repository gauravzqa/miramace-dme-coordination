# Execution Plan — Mira Mace DME Coordination Prototype

Built from two planning passes (core domain; AI + LiveKit voice), reconciled into one contract.
Demo: Tue Nov 3, 02:30 IST. Build window ≈ 20:15 → 01:15 IST, then verification, writeup, Mac smoke test.

---

## 0. Decisions (locked)

| Area | Decision |
|---|---|
| Language / tooling | Python 3.12, `uv`, Pydantic v2, `typer` + `rich`, `pytest` (+ `pytest-asyncio`) |
| Package / CLI | package `miramace`, CLI `mira` |
| DB | SQLite (stdlib), one file per run `runs/<case_id>.db`. Append-only `events` table is the source of truth; state is a pure projection |
| Concurrency model | `asyncio` throughout (voice is async). One case processed serially on a **virtual clock** |
| LLM | OpenAI Responses API, `responses.parse(text_format=PydanticModel)`. Text model from `MIRA_TEXT_MODEL` (user's choice, opaque id) |
| Voice | LiveKit Agents 1.8.10 against local `livekit-server --dev`. **Voice on the PCP leg only**; suppliers + companion are text over the same `CallerPort` contract |
| Other side of calls | LLM persona with a hidden fact sheet; reveals facts only when asked; behaviour cards (helpful, partial, ivr, transfer, callback_later, never_got_it, injection) |
| Offline-first | Build sandboxes cannot reach OpenAI. Every LLM path has a fake (`--llm fake`). Real runs (`--llm live`) and voice happen only on the Mac |
| Where it's built | Cloud workspace `/home/claude/miramace` (git repo), synced to `~/MiraMace/app` at each verified checkpoint |

## 1. Entities

| Entity | Key fields | Notes |
|---|---|---|
| `Case` | case_id, scenario, intake | One per run |
| `Patient` / `PatientProfile` | name, age, dob, mbi, coverage, address?, best_hours, caregiver, home_access | Address arrives via intake intent |
| `PCP` | name, practice, phone, npi | Dr. Chen |
| `Supplier` / `SupplierRecord` | sid, name, phone, address, distance_mi / status, attempts, facts, verified_at, ref, expires_at | status: UNCONTACTED·CALLING·NO_ANSWER·QUALIFIED·DISQUALIFIED·COMMITTED·RELEASED·UNREACHABLE |
| `SupplierFacts` | taking_new_medicare, k0001_in_stock, delivers_to_zip, accepts_assignment_all_months, earliest_delivery_days, contact_name, evidence{field→quote} | `None` = unknown |
| `SWO` | beneficiary_name, mbi, order_date, item_description, hcpcs, quantity, practitioner_name, practitioner_npi, signed, signature_date, accessories | Standard Written Order |
| `VisitNote` / `LCDChecklist` | doc_id, text, is_addendum / per-criterion met·not_met·not_documented + quote | 5 LCD criteria |
| `Commitment` | cid, party, what, due_at, status OPEN·KEPT·BROKEN·CANCELLED, followups | The ledger — "silence" = OPEN past due |
| `Task` | task_id, kind CALL_PCP·CALL_SUPPLIER·NOTIFY_PATIENT, party_id, intent, version, **idem_key**, due_at, attempts, max_attempts, status | Outbox; unique idem_key |
| `Event` | event_id, case_id, seq, type, payload, source{kind TIMER·TRANSCRIPT·PATIENT_INTENT·HUMAN·INBOUND_DOC·RULE·SYSTEM, ref}, causation_id, sim_time | Append-only |
| `Escalation` | eid, lane CASE·PATIENT, trigger, track, question, options[], suggested, context, status, resolution | Resolved via `HUMAN_RESOLVED` |
| `CaseState` | per-track `TrackState{state, owner, due_at, party_id}` + suppliers, swo, lcd, commitments, escalations… | Produced only by the reducer |
| `Snapshot` | headline, items[{track, plain_status, owner_label, due_label}], cost, allowed names/dates/amounts | Companion's only fact source |
| `PersonaSpec` | party_id, role, org_name, answerer_name, behaviour, facts[{id, slot, value, text, reveal_when}], sim{no_answer_attempts, ghost_after, lost_first_request, note/addendum/swo docs, delays} | Hidden from the caller |

## 2. Unified contracts (frozen in WP0 — everyone codes against these)

```python
# ---- calls ----
class SlotSpec(BaseModel): id: str; ask: str; type: Literal["bool","int","str","date"]; required: bool
class CallTask(BaseModel):
    task_id: str; party: Literal["pcp","supplier"]; party_id: str; org_name: str; phone: str
    intent: str; purpose: str; slots: list[SlotSpec]
    phi: PatientPHI | None            # None until code allows it (PHI gate)
    context: dict[str, str]; max_turns: int = 14
class Utterance(BaseModel):
    turn: int; speaker: Literal["caller","counterparty","system"]; name: str; text: str
    ts: float; heard_text: str | None = None; event: str | None = None   # hold/transfer/hangup
class Claim(BaseModel):
    field: str; status: Literal["stated","unknown"]; value: str | None; evidence_quote: str | None
class CallResult(BaseModel):
    task_id: str; outcome: Literal["COMPLETED","NO_ANSWER","VOICEMAIL","WRONG_NUMBER",
                                   "CALLBACK_REQUESTED","REFUSED","FAILED"]
    transcript: list[Utterance]; claims: list[Claim]; flags: list[str]
    duration_s: int; callback_after: datetime | None; transport: Literal["text","voice","text_fallback","fake"]
    usage: list[LLMCall]

# ---- patient ----
IntentKind = Literal["confirm_address","give_consent","ask_status","cost_concern","cost_accepted",
                     "confirm_delivery_slot","reschedule","needs_human","delivery_confirmed"]
class Intent(BaseModel): kind: IntentKind; address: str | None; preferred_time: str | None
                         note: str | None; utterance: str; conversation_id: str

# ---- ports (all async) ----
class LLMPort(Protocol):
    async def structured(self, *, agent: str, key: str, system: str, messages: list[dict], schema: type[T]) -> T
class CallerPort(Protocol):     async def run_call(self, task: CallTask) -> CallResult
class DocReaderPort(Protocol):  async def read_doc(self, doc: InboundDoc) -> list[Claim]     # visit note / SWO
class CompanionPort(Protocol):
    async def compose(self, req: ComposeRequest) -> str                                   # outbound update
    async def respond(self, snapshot: Snapshot, history: list[Utterance], text: str) -> CompanionReply
class PatientChannel(Protocol): async def deliver(self, case_id, purpose, text, at) -> PatientReply | None
class InboundPort(Protocol):    # fax/document simulator driven by PersonaSpec
    def observe(self, e: Event) -> None; def next_arrival(self, case_id) -> datetime | None
    def pop_due(self, case_id, now) -> list[InboundDoc]
class Brain(Protocol):          # shared by text and voice transports
    async def next(self, view: TranscriptView) -> CallerTurn | PersonaTurn
```

Value parsing ("Thursday" → datetime, "yes" → bool) happens in **code** (`interpret.py`) against the virtual clock.
Quote verification happens in the extractor layer **and** is re-checked in `interpret.py` (defence in depth).

## 3. Interactions

**Run loop (one step)**
```
events ─reduce→ CaseState ─invariants.check_all
       ─plan(state, now)→ desired TaskSpecs (+ rule-derived ESCALATION_OPENED / CASE_DONE)
       ─reconcile_tasks (idempotent by idem_key)
       ─advance virtual clock to min(next task due, next inbound doc)
       ─execute:  CALL_*  → build CallTask (PHI gate) → CallerPort.run_call → CallResult
                  NOTIFY  → notify policy → Companion.compose → grounding check → PatientChannel.deliver
                  INBOUND → DocReader.read_doc → claims
       ─interpret(task/doc, claims, state) → Events (rules decide) ─append → loop
```

**An outbound call (text or voice)**
```
CallTask → runner: persona answers → caller disclosure (template) → verify_party (code)
→ PHI unlocked only if verified → slot questions → readback → hangup
→ finalize_call: Extractor pass A + pass B → quote substring check → disagreement → unknown
→ CallResult → interpret → SUPPLIER_QUALIFIED / COMMITMENT_MADE / ORDER_REQUEST_VERIFIED …
```

**Patient chat**
```
Eleanor text → Companion.respond(snapshot, history) → grounding check (regenerate once / safe template)
→ keyword safety net (distress → needs_human) → IntentAPI.submit → validated → Events
```

**Escalation**
```
rule trigger → ESCALATION_OPENED (track owner = ADVOCATE, only that track blocks)
`mira resolve <eid> <option>` → HUMAN_RESOLVED + effect events → run loop resumes
```

## 4. Repo layout

```
miramace/
  pyproject.toml   README.md   .env.example   data/suppliers.csv   scenarios/*.yaml
  src/miramace/
    models.py events.py ports.py clock.py                # WP0 contracts
    store.py reducer.py invariants.py                     # core state
    rules.py geo.py planner.py escalation.py              # core decisions
    snapshot.py notify.py grounding.py intents.py         # patient side (code)
    executor.py interpret.py runloop.py scenario.py metrics.py cli.py
    llm/port.py (OpenAILLM, ScriptedLLM, ledger)
    agents/schemas.py agents/prompts.py
    calls/runner.py (Transcript, phases, disclosure, run_text_call, finalize_call, TextCallerPort)
    extract/extractor.py extract/lcd.py
    companion/agent.py
    fakes/ (FactSheetCaller, FactSheetDocReader, ScriptedPatient, InboundSimulator, KeywordCompanion, FakeBrains)
    voice/transport.py (VoiceCallerPort, two AgentSessions, listener link, text fallback)
  tests/
```

## 5. Work packages and waves

Each WP is owned by one builder agent, touches only its own files, and must pass its acceptance tests offline before handoff. After every wave the orchestrator runs an **independent verification agent** (reads the code cold, runs the full test suite, checks the WP against this plan) before the next wave starts.

### Wave 0 — contracts (1 builder, ~30 min) · critical path
| WP | Scope | Acceptance |
|---|---|---|
| **WP0** | pyproject (uv; extras `voice`), `models.py`, `events.py` (EventType enum + payload registry), `ports.py`, `clock.py` (business hours CT, patient hours), `agents/schemas.py` (all LLM I/O models), data copy, test scaffold | `uv run pytest` green; Fri 16:00 + 2 bh = Mon 10:00; every LLM schema passes `to_strict_json_schema` |

### Wave 1 — five builders in parallel (~45 min)
| WP | Scope | Acceptance |
|---|---|---|
| **B1 State** | `store.py`, `reducer.py` (all transition tables, derived supplier track, ledger, escalation effects), `invariants.py` | Append/reload round-trip; duplicate idem_key ignored; illegal transition raises; each invariant trips on a hand-built bad state |
| **B2 Decisions** | `rules.py`, `geo.py`, `planner.py`, `escalation.py` | From 60614 nearest = Lakeshore / Lincoln Park; K0003 → DRIFT; SWO missing NPI detected; follow-ups at 24/48/72 bh; one screening call in flight |
| **B3 AI calls** | `llm/port.py` (OpenAILLM, ScriptedLLM, ledger), `agents/prompts.py`, `calls/runner.py`, `extract/extractor.py`, `extract/lcd.py`, FakeBrains (rule-based caller/persona) | Helpful Maria completes ≤14 turns; transfer → disclosure twice; pre-verify prompt contains no PHI; fabricated quote → unknown; injection sentence never used as evidence |
| **B4 Patient side** | `snapshot.py`, `notify.py`, `grounding.py`, `intents.py`, `companion/agent.py`, fakes for patient + companion | Invented "$150"/"Friday" held; "I'm scared" → needs_human even if LLM omits it; cost_concern opens PATIENT-lane escalation; ≤1 notify/day |
| **B5 Voice** | `voice/transport.py`: two in-process `AgentSession`s on local LiveKit, pinned identities, `participant_kinds` incl. AGENT, VAD turn detection, interruptions off, watchdog, listener token/link, **automatic text fallback** | Offline: ImportError / connection failure → text fallback produces a valid CallResult; BrainAgent logic unit-tested with a stub. Live: Mac smoke only |

### Wave 2 — integration (2 builders, ~50 min)
| WP | Scope | Acceptance |
|---|---|---|
| **B6 Engine** | `executor.py` (CallTask build + PHI gate), `interpret.py` (claims → events, value parsing, re-check quotes), `fakes/` FactSheetCaller + InboundSimulator + FactSheetDocReader, `runloop.py`, `scenario.py`, `metrics.py`, `scenarios/happy_path.yaml` | happy_path → DONE, 0 escalations, invariants hold every step, < 2 s with fakes |
| **B7 Scenarios + CLI** | 6 more scenarios, `cli.py` (`run`, `run-all`, `timeline`, `snapshot`, `chat`, `escalations`, `resolve`, `call pcp [--voice --listen]`) | `run-all`: 0 false closes, escalation recall 1.0; `resolve` on cost_concern resumes to DONE |

Scenarios (7): `happy_path`, `supplier_disqualified` (out of stock + no assignment), `ghost_failover` (yes-then-silent → backup + ORDER_REDIRECT), `pcp_never_got_it` (verify-first resend), `lcd_missing_addendum` (addendum fixes it), `cost_concern` (patient lane → nurse → resolve), `code_drift_k0003` (escalate).

### Wave 3 — verify, document, hand off (~45 min)
| WP | Scope |
|---|---|
| **V-final** | Independent verifier: full test suite, `run-all`, code review against architecture (AI never mutates state, PHI gate, idempotency), list of defects → fix loop |
| **Docs** | README (setup, `.env`, run commands, demo script), 1–2 page writeup (sequencing, architecture, AI in/out, cut list, what's next), sync to `~/MiraMace/app`, push-ready git repo |
| **Mac smoke** (you) | see §7 |

**Critical path:** WP0 → B1/B2 → B6 → B7 → V-final ≈ 3h 20m. B3/B4/B5 run alongside.

## 6. Verification protocol (after every task)

1. Builder runs its acceptance tests and reports files touched.
2. Orchestrator runs the full suite + lint (`ruff`) in the shared repo.
3. A separate verifier agent (did not write the code) checks the WP against this plan and the architecture invariants, and tries one adversarial case. Defects go back to the builder.
4. Only then is the wave marked done and synced to the Mac.

## 7. Mac smoke checklist (needs your OpenAI key)

```
brew install livekit livekit-cli
livekit-server --dev                         # ws://localhost:7880  devkey / secret
cd ~/MiraMace/app && uv sync --extra voice && cp .env.example .env   # add OPENAI_API_KEY, MIRA_TEXT_MODEL
uv run mira run-all --llm fake               # offline sanity
uv run mira run happy_path --llm live        # real LLM, text calls
uv run mira call pcp --voice --listen        # opens Chrome Meet link; hear caller + Maria
uv run mira run happy_path --llm live --voice-pcp
uv run mira chat <case_id> --llm live        # talk as Eleanor
```

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agents don't hear each other (AGENT kind filtered by default) | `participant_kinds` incl. AGENT + `participant_identity` pinning; first Mac task is a 10-min voice spike |
| Default turn detector / interruptions reach LiveKit Cloud | `turn_detection="vad"`, interruptions disabled |
| Text model latency per turn | `MIRA_TURN_MODEL` override, low reasoning effort, 2-sentence turns |
| Silence / endless loop | max_turns, 20 s watchdog, 240 s cap |
| Voice fails live | automatic text fallback per call; demo continues |
| No OpenAI in build sandboxes | fakes everywhere; live verified only on Mac |
| Time | cut order below |

## 9. Cut order if time runs short

1. Console-mode voice (you playing Maria) — cut already.
2. Replay cassettes.
3. Scenarios beyond 5 (drop `code_drift_k0003`, then `supplier_disqualified`).
4. Companion live chat → keep scripted patient only.
5. Voice → text-only PCP leg with voice shown as the 1-day next step.

## 10. Deviations from architecture.md (to note in writeup)

- Order validates **before** routing (`SIGNED → VALID → SENT_TO → RECEIVED_BY`) so a bad order never reaches a supplier.
- Added intents `cost_accepted`, `confirm_delivery_slot`, `delivery_confirmed`.
- Supplier lease 72 h; re-verify within 24 h of scheduling; screening stops at primary + backup.
- Distance = ZIP-centroid heuristic; the supplier's "delivers to ZIP" answer is authoritative.
- Business hours 9–17 CT weekdays, no holiday calendar; sim starts Mon 2026-11-02 10:00 CT.
