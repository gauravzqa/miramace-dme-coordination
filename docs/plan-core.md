# Core (non-LLM) build plan — reference for builders

Authoritative contract decisions live in `execution-plan.md` §2 (unified contracts). Where this file
differs (package `miramace` sync ports, Claim shape, intent names), **execution-plan.md wins**:
ports are async; `Claim{field, status, value:str|None, evidence_quote}`; intents are lower snake
(`confirm_address, give_consent, ask_status, cost_concern, cost_accepted, confirm_delivery_slot,
reschedule, needs_human, delivery_confirmed`). Persona format = `PersonaSpec` (execution-plan §1).

## Module responsibilities
- models.py — all Pydantic entities and enums
- events.py — EventType enum, payload models, PAYLOADS registry, make_event()
- store.py — SQLite: append/load events, task reconcile, transcripts, projection-cache sync
- clock.py — VirtualClock, business-hours + quiet-hours math
- reducer.py — pure: reduce(state, event) -> state; reduce_all(events)
- rules.py — pure domain rules
- geo.py — ZIP centroid table + distance heuristic
- planner.py — pure: plan(state, now, policy) -> PlannerDecision(tasks, events)
- interpret.py — pure: (task|doc, CallResult|claims, state) -> list[Event]; value parsing; quote re-check
- executor.py — impure: build CallTask (PHI gate), run via ports, return Events
- snapshot.py, notify.py, grounding.py, intents.py — patient side code
- escalation.py — trigger catalogue, option -> effect table
- invariants.py — 5 invariants as assert functions
- runloop.py, scenario.py, metrics.py, cli.py

## Entities (fields)
Use StrEnum; tz-aware datetimes in America/Chicago.
```
Owner: us, pcp, supplier, patient, advocate, none
TrackState{state:str, owner:Owner, due_at:datetime|None, party_id:str|None}
Address{line1, city, state, zip}
PatientProfile{best_hours:(9,20), caregiver_name?, caregiver_phone?, hearing_or_language?, prior_concerns:[], home_access?}
Patient{name, age, mbi="1EG4-TE5-MK72", dob, coverage="ORIGINAL_PART_B", medigap=False, address?, profile}
PCP{name, practice, phone, npi="1234567893"}
Supplier{sid (slug of name), name, phone, address, distance_mi?}
SupplierFacts{taking_new_medicare?, k0001_in_stock?, delivers_to_zip?, accepts_assignment_all_months?, earliest_delivery_days?, contact_name?, evidence{field->quote}}
SupplierStatus: UNCONTACTED CALLING NO_ANSWER QUALIFIED DISQUALIFIED COMMITTED RELEASED UNREACHABLE
SupplierRecord{sid, status, attempts=0, facts?, verified_at?, ref?, expires_at?, reason?}
SWO{beneficiary_name?, mbi?, order_date?, item_description?, hcpcs?, quantity?, practitioner_name?, practitioner_npi?, signed=False, signature_date?, accessories=[]}
LCDCriterion: MRADL_LIMITATION CANE_WALKER_INSUFFICIENT HOME_ACCOMMODATES WILL_USE_IMPROVES SELF_PROPEL_OR_CAREGIVER
LCDChecklist{results{criterion->"met"|"not_met"|"not_documented"}, evidence{criterion->quote}}
VisitNote{doc_id, text, received_at, is_addendum=False}
Commitment{cid, party, what, due_at, status OPEN|KEPT|BROKEN|CANCELLED, source_event_id, followups=0}
EscalationLane: CASE PATIENT
Escalation{eid, lane, trigger, track, question, options[{option_id,label}], suggested, context, status OPEN|RESOLVED, resolution?}
TaskKind: CALL_PCP CALL_SUPPLIER NOTIFY_PATIENT
Task{task_id, case_id, kind, party_id, intent, version, idem_key, due_at, attempts=0, max_attempts, status PENDING|DONE|CANCELLED|FAILED, args{}}
TaskSpec = Task minus ids/status
CaseState{case_id; case, patient, order, supplier, assignment, delivery: TrackState; patient_info; pcp;
  suppliers{sid->SupplierRecord}; committed_sid?; backup_sid?; swo?; lcd?; addendum_requests=0; correction_requests=0;
  order_request_version=0; order_sent_to?; order_received_by?; delivery_slot?; commitments{}; escalations{};
  last_notified_at?; notify_attempts=0; pending_notify[]; patient_flags(set of seen: address, consent); resume_to?; last_seq}
SnapshotItem{track, plain_status, owner_label, due_label?}
CostInfo{deductible_usd=283, coinsurance_pct=20, rental_months=13}
Snapshot{case_id, as_of, headline, items[], cost, next_patient_action?, allowed_names, allowed_dates, allowed_amounts}
```

## SQLite schema
```sql
CREATE TABLE cases(case_id TEXT PRIMARY KEY, scenario TEXT, intake_json TEXT NOT NULL, created_sim TEXT NOT NULL);
CREATE TABLE events(event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
  payload_json TEXT NOT NULL, source_kind TEXT NOT NULL, source_ref TEXT NOT NULL, causation_id TEXT,
  sim_time TEXT NOT NULL, recorded_at TEXT NOT NULL, UNIQUE(case_id, seq));
CREATE INDEX ix_events_case_type ON events(case_id, type);
CREATE TABLE tasks(task_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, kind TEXT, party_id TEXT, intent TEXT, version INT,
  idem_key TEXT NOT NULL UNIQUE, due_at TEXT NOT NULL, attempts INT DEFAULT 0, max_attempts INT, status TEXT, args_json TEXT, last_result TEXT);
CREATE INDEX ix_tasks_due ON tasks(case_id, status, due_at);
CREATE TABLE transcripts(transcript_id TEXT PRIMARY KEY, case_id TEXT, task_id TEXT, party_id TEXT, outcome TEXT,
  turns_json TEXT, claims_json TEXT, flags_json TEXT, transport TEXT, sim_time TEXT);
CREATE TABLE suppliers(sid TEXT PRIMARY KEY, name TEXT, phone TEXT, address_json TEXT, zip TEXT);
CREATE TABLE commitments(cid TEXT PRIMARY KEY, case_id TEXT, party TEXT, what TEXT, due_at TEXT, status TEXT, source_event_id TEXT);
CREATE TABLE escalations(eid TEXT PRIMARY KEY, case_id TEXT, lane TEXT, trigger TEXT, track TEXT, status TEXT, body_json TEXT, resolution TEXT);
CREATE INDEX ix_esc_open ON escalations(status, lane);
```
events append-only (no update/delete API). tasks = outbox; UNIQUE idem_key. WAL. DB per run `runs/<case_id>.db`.
commitments/escalations tables are rebuildable caches written from CaseState after each append; core never reads them.

## Event catalogue
Envelope: `Event(event_id, case_id, seq, type, payload, source{kind, ref, span?}, causation_id?, sim_time)`.
Source.kind: TIMER TRANSCRIPT PATIENT_INTENT HUMAN INBOUND_DOC RULE SYSTEM. `PAYLOADS[type]` validates payload on append.

| Track | Events (payload) |
|---|---|
| Case | CASE_OPENED(patient, pcp), CASE_PAUSED(reason), CASE_RESUMED, CASE_CANCELLED(reason), COVERAGE_CHANGED(new), CASE_DONE |
| Patient | PATIENT_NOTIFIED(purpose, text, grounding_ok, answered), ADDRESS_CONFIRMED(address, home_access?), CONSENT_GIVEN(scope), COST_DISCLOSED(cost), COST_ACCEPTED, CONCERN_RAISED(kind cost/confusion/distress/clinical/human, utterance), STATUS_ASKED(utterance), PATIENT_UNREACHABLE |
| Order | SWO_DRAFTED(swo), ORDER_REQUESTED(request_ref, version), ORDER_REQUEST_VERIFIED(received, named_owner?), VISIT_NOTE_RECEIVED(note), LCD_EVALUATED(checklist, missing[]), ADDENDUM_REQUESTED(missing[]), SWO_SIGNED_RECEIVED(swo), SWO_EVALUATED(complete, missing[], hcpcs), CORRECTION_REQUESTED(fields[]), ORDER_VALIDATED, ORDER_SENT(to_sid), ORDER_RECEIVED(by_sid), ORDER_REDIRECT(from_sid, to_sid), ORDER_STALLED(followups), PCP_REFUSED(reason) |
| Supplier | CANDIDATES_RANKED(sids[], distances), SUPPLIER_NO_ANSWER(sid, attempt), SUPPLIER_QUALIFIED(sid, facts, verified_at), SUPPLIER_DISQUALIFIED(sid, reason), SUPPLIER_COMMITTED(sid, ref, expires_at), SUPPLIER_REVERIFIED(sid, verified_at), SUPPLIER_RELEASED(sid, reason), SUPPLIERS_EXHAUSTED |
| Assignment | ASSIGNMENT_CONFIRMED(sid), ASSIGNMENT_REFUSED(sid) |
| Delivery | DELIVERY_SLOT_PROPOSED(sid, slot), DELIVERY_SCHEDULED(sid, slot), DELIVERY_MISSED, DELIVERED(sid, at), POD_CONFIRMED |
| Ledger | COMMITMENT_MADE(cid, party, what, due_at, evidence), COMMITMENT_KEPT(cid), COMMITMENT_BROKEN(cid) |
| Audit | CALL_COMPLETED(task_id, transcript_id, outcome), EXTRACTION_FAILED(task_id, field, reason) |
| Escalation | ESCALATION_OPENED(escalation), HUMAN_RESOLVED(eid, option_id, note, resolver) |

## State machines (reducer.py) — (state, event[guard]) -> new state; owner; due
Guard failure → IllegalTransition; store.append rejects.

**Case**: ACTIVE —CASE_PAUSED→ PAUSED (release leases) —CASE_RESUMED→ ACTIVE; any —CASE_CANCELLED→ CANCELLED; any —COVERAGE_CHANGED→ COVERAGE_CHANGED; ACTIVE —CASE_DONE[delivery=POD_CONFIRMED]→ DONE.

**Patient** (INTAKE, owner patient): INTAKE —ADDRESS_CONFIRMED & CONSENT_GIVEN (both seen)→ CONSENTED (owner us); CONSENTED —COST_DISCLOSED→ COST_DISCLOSED (owner patient, due now+24bh); COST_DISCLOSED —COST_ACCEPTED→ COST_ACCEPTED; any —CONCERN_RAISED→ CONCERN_RAISED (owner advocate, resume_to saved); CONCERN_RAISED —HUMAN_RESOLVED→ per option table; COST_ACCEPTED —POD_CONFIRMED→ DELIVERY_CONFIRMED.

**Order**: NOT_SENT —ORDER_REQUESTED→ REQUESTED (owner pcp, due=commitment due); REQUESTED —VISIT_NOTE_RECEIVED→ DOCS_CHECK (owner us); DOCS_CHECK —LCD_EVALUATED[missing≠∅]→ DOCS_CHECK; DOCS_CHECK —ADDENDUM_REQUESTED→ ADDENDUM_REQUESTED (owner pcp); ADDENDUM_REQUESTED —VISIT_NOTE_RECEIVED→ DOCS_CHECK; DOCS_CHECK|REQUESTED —SWO_SIGNED_RECEIVED→ SIGNED; SIGNED —CORRECTION_REQUESTED→ CORRECTION_REQUESTED (owner pcp); CORRECTION_REQUESTED —SWO_SIGNED_RECEIVED→ SIGNED; SIGNED —ORDER_VALIDATED[LCD all met ∧ SWO complete ∧ hcpcs=K0001]→ VALID; VALID —ORDER_SENT(sid)[sid=committed_sid]→ SENT_TO (owner supplier); SENT_TO —ORDER_RECEIVED→ RECEIVED_BY; SENT_TO|RECEIVED_BY —ORDER_REDIRECT→ VALID (clear order_sent_to/received_by); open state —ORDER_STALLED|PCP_REFUSED|ESCALATION_OPENED(track=order)→ BLOCKED_ON_HUMAN.
Note: LCD may complete (all met) while still awaiting the signed SWO; record checklist in state.

**Supplier**: per-record UNCONTACTED→NO_ANSWER(n)→QUALIFIED|DISQUALIFIED|UNREACHABLE. Track state derived: ∃COMMITTED→COMMITTED(sid) else ∃QUALIFIED→QUALIFIED else any uncontacted/retryable→CALLING else EXHAUSTED. SUPPLIER_COMMITTED[no other COMMITTED ∧ assignment confirmed for sid] → COMMITTED (owner supplier, due=expires_at). SUPPLIER_REVERIFIED refreshes verified_at. SUPPLIER_RELEASED → record RELEASED, track re-derives (backup).

**Assignment** per supplier: UNKNOWN → CONFIRMED_ALL_MONTHS | REFUSED; track reflects committed/best.

**Delivery**: NOT_STARTED —DELIVERY_SLOT_PROPOSED→ SCHEDULING (owner patient); SCHEDULING —DELIVERY_SCHEDULED[can_schedule]→ SCHEDULED (owner supplier, due=slot+4h); SCHEDULED —DELIVERED→ DELIVERED (owner patient); DELIVERED —POD_CONFIRMED→ POD_CONFIRMED; SCHEDULED —DELIVERY_MISSED→ SCHEDULING.

**Ledger**: COMMITMENT_MADE→OPEN; VISIT_NOTE_RECEIVED / SWO_SIGNED_RECEIVED / ORDER_RECEIVED / DELIVERED close matching party commitments as KEPT.
**Escalation**: ESCALATION_OPENED sets that track's owner=advocate; HUMAN_RESOLVED applies escalation.EFFECTS[(trigger, option_id)] (follow-up events appended by run loop with source HUMAN).

## Planner (pure) — plan(state, now, policy) -> PlannerDecision(tasks: list[TaskSpec], events: list[Event])
idem_key = f"{case_id}:{party_id}:{intent}:v{version}". store.reconcile_tasks inserts new keys, cancels PENDING no-longer-desired, never touches DONE. Case not ACTIVE → nothing. Track with owner advocate → nothing for it.
1. Patient INTAKE & no welcome → NOTIFY_PATIENT(welcome_intake) due next_patient_time; 3 attempts → caregiver → escalate PATIENT_UNREACHABLE.
2. Patient CONSENTED+ & order NOT_SENT → CALL_PCP(request_order, version=order_request_version+1).
3. Open PCP commitment past due → CALL_PCP(followup_order_k) at due+24/48/72bh (k=1..3); follow-up verifies first; ORDER_REQUEST_VERIFIED(received=False) → version bump → resend legal. k>3 → escalate ORDER_STALLED.
4. LCD missing: addendum_requests==0 → CALL_PCP(request_addendum, args missing) else escalate LCD_MISSING_AFTER_ADDENDUM.
5. SWO_EVALUATED hcpcs∉{K0001} → escalate CODE_DRIFT; incomplete & correction_requests==0 → CALL_PCP(request_correction) else escalate.
6. Patient address known & <2 QUALIFIED/COMMITTED → CALL_SUPPLIER(screen) for next-ranked UNCONTACTED or NO_ANSWER with attempts<3 (retry due last+2bh). One screening call in flight. None left & none qualified → escalate SUPPLIERS_EXHAUSTED.
7. No COMMITTED & some QUALIFIED with assignment → CALL_SUPPLIER(commit, best). PHI shared here after verification. Lease 72h.
8. COMMITTED & patient CONSENTED → NOTIFY_PATIENT(cost_disclosure).
9. Order VALID & COMMITTED → CALL_PCP(send_order_<sid>, args to_sid).
10. Order SENT_TO(sid) past PCP commitment due → CALL_SUPPLIER(sid, verify_order_received).
11. RECEIVED_BY ∧ COST_ACCEPTED ∧ (now - verified_at > 24h) → CALL_SUPPLIER(reverify); fresh → CALL_SUPPLIER(propose_slot); slot proposed → NOTIFY_PATIENT(delivery_slot).
12. Lease expired or reverify failed twice → SUPPLIER_RELEASED (+ ORDER_REDIRECT if order already sent) → rule 7 commits backup → rule 9.
13. Delivery SCHEDULED past due & no DELIVERED → CALL_SUPPLIER(verify_delivered).
14. DELIVERED → NOTIFY_PATIENT(delivered_checkin); delivery_confirmed → POD_CONFIRMED → CASE_DONE (planner emits CASE_DONE when predicate holds).
Retry budgets: supplier screen 3 attempts / 2bh apart; reverify 2; PCP 3 follow-ups 24/48/72bh; patient 3 in best hours then caregiver; notify ≤1 per sim day except action-required (cost_disclosure, delivery_slot, welcome_intake).

## Rules (pure)
```
rank_suppliers(suppliers, patient_zip) -> [(sid, miles)]  # asc distance; tie by name
supplier_eligible(f) -> (bool, reason|None)   # all four booleans True; None => "unknown:<field>" (one re-screen then DISQUALIFIED)
facts_consistent(f) -> [issues]               # e.g. in_stock False and earliest_delivery_days <= 3
swo_completeness(swo) -> [missing]            # 7 SWO elements + signed + signature_date
lcd_evaluate(claims) -> LCDChecklist          # met only if claim stated true AND evidence present
check_code(hcpcs) -> "OK"|"DRIFT"|"MISSING"   # DRIFT for K0002-K0007 or any non-K0001
assignment_ok(sid, state) -> bool
can_schedule(state, now) -> (bool, [reasons])  # invariant 1 predicate
cost_info() -> CostInfo                         # constants only; never invent an allowed $ amount
lcd_missing_wording(criterion) -> str
```
geo.ZIP_CENTROIDS: hard-coded lat/lon for directory ZIPs (60614, 60654, 60618, 60640, 60605, 60608, 60657, 60714, 60201) + patient ZIP; haversine; unknown ZIP fallback abs(int(z1)-int(z2))*0.5 mi.

## Clock & run loop
VirtualClock now()/advance_to(t) monotonic; start 2026-11-02 10:00 America/Chicago. is_business (Mon-Fri 9-17 CT), next_business_time, add_business_hours, next_patient_time(t, profile).
```
loop:
  state = reduce_all(events); invariants.check_all(state, tasks, now)
  dec = plan(state, now, policy); append dec.events (source RULE); reconcile_tasks(dec.tasks)
  t_task = next_pending_due; t_in = inbound.next_arrival
  if neither: break -> DONE | BLOCKED_ON_HUMAN | STUCK (STUCK = invariant-2 violation)
  clock.advance_to(max(now, min(t_task, t_in)))
  inbound first: for doc in pop_due -> executor.ingest_doc(doc)
  else: events = executor.execute(task)  # port call, store transcript, interpret -> events
  for e in events: append(e); inbound.observe(e)
```
Calls advance clock by duration_s. Tasks due outside allowed hours are shifted by reconcile (invariant 5).

Escalation triggers + options: LCD_MISSING_AFTER_ADDENDUM{pcp_sent_addendum, cancel}; ORDER_STALLED{advocate_called_got_owner, switch_to_portal}; CODE_DRIFT{pcp_corrected_to_K0001 (emits corrected SWO_SIGNED_RECEIVED), proceed_new_code_manual (blocks), cancel}; SUPPLIERS_EXHAUSTED{expand_radius, advocate_found_supplier}; EXTRACTION_CHECK_FAILED{accept_value, recall}; COST_CONCERN{nurse_resolved_accepted → COST_ACCEPTED, patient_declined → CASE_CANCELLED}; PATIENT_UNREACHABLE / NEEDS_HUMAN / PCP_REFUSED similar.

## Invariants
- assert_single_commit: ≤1 COMMITTED.
- assert_schedule_preconditions: delivery ∈ {SCHEDULED, DELIVERED, POD_CONFIRMED} ⇒ order_received_by==committed_sid, SWO valid K0001, LCD all met, assignment confirmed, verified_at within 24h of DELIVERY_SCHEDULED event time, patient COST_ACCEPTED.
- assert_liveness: non-terminal ⇒ PENDING task or open escalation or pending inbound arrival; terminal ⇒ no pending tasks.
- assert_idempotent_outbound: unique idem keys; request_order v>1 only after ORDER_REQUEST_VERIFIED(received=False).
- assert_traceable: every event has source kind+ref; TRANSCRIPT events reference an existing transcript.
- assert_patient_contact_ok: every PATIENT_NOTIFIED grounding_ok=True & within patient hours; every call within business hours.

## Scenario harness
YAML: name; patient{address, best_hours}; patient_script{purpose -> [intents]}; personas{party_id -> PersonaSpec}; ground_truth{claim_would_pass, lcd_supported, hcpcs}; expect{final DONE|BLOCKED_ON_HUMAN, escalations[], max_sim_days}.
Fakes: FactSheetCaller (deterministic templated transcript from persona facts; honours no_answer_attempts, ghost_after, lost_first_request) + extraction from the templated lines (exact answer line as quote); FactSheetDocReader (scripted claims for notes/SWO, quotes must appear in doc text); ScriptedPatient; InboundSimulator (observe ORDER_REQUESTED → schedule visit note fax after note_delay_bh, signed SWO after signed_delay_bh, etc.); KeywordCompanion.
Metrics: autonomous_close (DONE ∧ 0 escalations); false_close (reached SCHEDULED+ while oracle on fact-sheet truth says claim fails) — hard gate, CLI exit 1; escalation precision/recall vs expect; per scenario sim days, call count, invariant violations (0), PHI-before-qualification (0).

## CLI
run <scenario> [--llm fake|live] [--voice-pcp] [--db]; run-all; timeline <case_id> [--track]; snapshot <case_id>; chat <case_id>; escalations [--open]; resolve <eid> <option> --note --by; transcript <id>; call pcp [--voice --listen].
