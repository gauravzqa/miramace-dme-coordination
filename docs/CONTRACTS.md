# Frozen contracts (WP0) — index

Authoritative over plan docs for *names*; `execution-plan.md` §2 remains authoritative for semantics.
Changing anything here requires orchestrator sign-off.

| Module | Key exports |
|---|---|
| `models.py` | `CHICAGO`, `PCP_PARTY_ID="pcp"`, `TARGET_HCPCS`, `idem_key()`, `slugify()`; enums `Owner Track CaseStatus PatientStatus OrderStatus SupplierTrackStatus AssignmentStatus DeliveryStatus SupplierStatus LCDCriterion CommitmentStatus EscalationLane EscalationStatus EscalationTrigger TaskKind TaskStatus InboundDocKind`; Literals `IntentKind NotifyPurpose CallOutcome CallTransport LCDResult PersonaBehaviour`; `LCD_CRITERION_BY_LLM_ID`; models `TrackState Address PatientProfile PatientPHI Patient PCP Supplier SupplierFacts SupplierRecord SWO LCDChecklist VisitNote InboundDoc Commitment EscalationOption Escalation TaskSpec Task SlotSpec CallTask Utterance Claim LLMCall CallResult Intent SnapshotItem CostInfo Snapshot ComposeRequest CompanionReply PatientReply Fact PersonaSim PersonaSpec CaseState`; `eleanor() dr_chen() load_suppliers()` |
| `events.py` | `EventType` (51 types), one payload model per type, `PAYLOADS`, `Source`, `SourceKind`, `Event` (+`.parsed()`), `make_event()`, `PlannerDecision` |
| `ports.py` | `LLMPort CallerPort DocReaderPort CompanionPort PatientChannel InboundPort Brain` (runtime_checkable Protocols), `TranscriptView`, `CallPhase`, `Ports` dataclass |
| `clock.py` | `SIM_START`, `VirtualClock(now, advance_to, advance)`, `ClockError`, `is_business`, `next_business_time`, `add_business_hours`, `next_patient_time`, `is_patient_time` |
| `agents/schemas.py` | `CallerTurn PersonaTurn ClaimLLM Extraction CriterionResult LCDCheck IntentLLM CompanionOut`, `LLM_SCHEMAS` |

## Conventions
- **Ports are async** (`await`), except `InboundPort` (sync simulator). Implement them structurally; no base classes.
- **idem_key** = `models.idem_key(case_id, party_id, intent, version)` → `f"{case_id}:{party_id}:{intent}:v{version}"`.
- **party_id**: PCP = `"pcp"`; supplier = its `sid` (slug of name, e.g. `lakeshore-home-medical-equipment`).
- **Time**: all datetimes tz-aware, America/Chicago (`CHICAGO`). `Event.sim_time` rejects naive datetimes.
  Business hours Mon–Fri 09:00–17:00; `add_business_hours` never returns 17:00 (rolls to next 09:00).
  Patient hours = `profile.best_hours` (default 9–20), any day.
- **Intents** are lower snake (`IntentKind`). Event types / statuses are UPPER_SNAKE StrEnums; `Owner` values lower.
- **Events**: build only via `make_event()` (validates against `PAYLOADS`, payload models are `extra="forbid"`).
  `Event.payload` is a JSON-mode dict (datetimes are ISO strings) → use `event.parsed()` for typed access.
  `Event.seq` is 0 until **store.append assigns it**. Payload-less events use `None`/`{}` (model `Empty`).
- **CaseState** is produced only by `reducer.py`. `TrackState.state` is `str` (values from the status enums).
  WP0 additions to plan-core: `SupplierTrackStatus.NOT_STARTED` (before ranking), `ranked`, `assignment_by_sid`,
  `swo_missing`, `visit_notes` on CaseState; `CaseState.track(name)` accessor.
- **LLM schemas** (agents/schemas.py): every field required, optional = `X | None` with no default, no dicts.
  Convert to domain in code: `ClaimLLM→Claim`, `IntentLLM→Intent` (add utterance, conversation_id),
  `CriterionResult.criterion_id→LCDCriterion` via `LCD_CRITERION_BY_LLM_ID`.
- **Choices recorded**: `CANDIDATES_RANKED.distances` is `dict[sid, miles]`; `EscalationTrigger` enum includes
  `SWO_INCOMPLETE_AFTER_CORRECTION`; `PatientReply{conversation_id, text, intents, at}`; `CompanionReply{reply,
  intents, grounding_ok, flags}`; `ComposeRequest{case_id, purpose, snapshot, profile, history}`;
  `LLMCall` lives in models (CallResult.usage needs it); `Ports.llm` optional (None in pure-fake runs).
- All `livekit` imports stay inside `voice/`. Never call OpenAI in tests.
