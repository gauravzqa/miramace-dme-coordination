# Live defense run sheet (45 min)

Setup before the call: two terminals at ~160 columns (`export COLUMNS=160`), `livekit-server --dev` running in a third, `.env` filled, Chrome open. Do a `uv run mira run-all --llm fake` dry run 10 minutes before. Have `docs/WRITEUP.md` and `docs/architecture.md` open in tabs.

| Time | Block |
|---|---|
| 0–5 | Framing |
| 5–10 | happy_path timeline |
| 10–14 | ghost_failover |
| 14–17 | pcp_never_got_it |
| 17–22 | cost_concern → resolve |
| 22–26 | companion chat |
| 26–31 | voice PCP call |
| 31–33 | run-all metrics |
| 33–45 | Questions |

## 0–5 · Framing

- The claim dies on paperwork, not on logistics: written order + LCD medical-necessity note + assignment every rental month. So the PCP order leg is built deep, the supplier leg broad and thin.
- **AI owns the conversations, code owns the commitments.** LLMs talk and extract quoted claims; a pure reducer and planner over an event log decide everything.
- A human comes back only on a closed set of triggers, and only the blocked track waits.

## 5–10 · happy_path

```bash
uv run mira run happy_path
```
Point at:
- `CANDIDATES_RANKED` (RULE) → `SUPPLIER_QUALIFIED` (TRANSCRIPT) with the **evidence quote** inline. The source column says *why* each line happened.
- `COMMITMENT_MADE` "Dr. Chen's office will fax … by Thu Nov 5 17:00 'By Thursday.'" — the ledger. "Thursday" was parsed by code against the virtual clock, not by the LLM.
- `LCD_EVALUATED` all 5 criteria met → `ORDER_VALIDATED` (RULE) **before** `ORDER_SENT` to the committed supplier.
- `SUPPLIER_REVERIFIED` Thu 09:01, within 24h of scheduling (invariant 1).
- Summary: 4.0 sim days, 8 calls, invariants all held, PHI before qualification 0.

Optional: `uv run mira transcript <tr_id> --case <case_id>` (ids from `mira transcripts <case_id>`) to show claims with quotes.

## 10–14 · ghost_failover

```bash
uv run mira run ghost_failover
```
Point at Thursday: `SUPPLIER_NO_ANSWER` ×2 → `SUPPLIER_RELEASED` (RULE: "no answer to 2 re-verification calls") → `ORDER_REDIRECT` Lakeshore → Lincoln Park → `SUPPLIER_COMMITTED` Lincoln → `ORDER_SENT` → delivered Fri. Same 4.0 days as the happy path; no human. The burst of no-answers after it is the system looking for a *new* backup (23 calls total).

## 14–17 · pcp_never_got_it

```bash
uv run mira run pcp_never_got_it
```
Point at `ORDER_REQUEST_VERIFIED` "PCP says request NOT received -> resend" then `ORDER_REQUESTED … :v2`. **Verify first, then resend** under a new idempotency key — never a blind duplicate. Honest note: 15 sim days, because the first follow-up waits 24 business hours after the missed Thursday promise. That grace period is a policy knob; an advocate would likely call Friday morning.

## 17–22 · cost_concern → resolve → DONE

```bash
uv run mira run cost_concern                    # ends BLOCKED_ON_HUMAN
uv run mira escalations <case_id> --open
uv run mira resolve <case_id> esc-cost_concern-1 nurse_resolved_accepted --note "Explained 20% share"
```
Point at: `CONCERN_RAISED` with her exact words; the escalation is **PATIENT lane**, with typed options and a suggested one; the order track kept moving (visit note arrived Wed while blocked). After resolve: `HUMAN_RESOLVED` → delivered → `CASE_DONE`. The resolve summary prints `expectation FAIL` because the scenario's expectation is the pre-resolve state; ignore it and say so.

## 22–26 · Companion chat

```bash
uv run mira chat <cost_concern case_id> --llm live     # fallback: --llm fake
```
Type, in order:
1. `Where is my wheelchair?` → answered from the snapshot only. `/snapshot` to show its entire fact source.
2. `Will it cost me $150?` → no invented amount (grounding check: every $/date/name must be in the snapshot).
3. `I'm scared, can I talk to a real person?` → `flags: safety_net_needs_human`, `intent needs_human: accepted -> CONCERN_RAISED, ESCALATION_OPENED`. The keyword net fires even if the model misses it.

## 26–31 · Voice PCP call

```bash
uv run mira call pcp --voice --listen
```
Open the printed `meet.livekit.io` link in Chrome, mic/camera off, Join. Listen for: AI disclosure on the first turn, practice verification before any PHI, one question at a time, readback, hangup. Then the table: STT text vs `heard_text`, claims with quotes, `transport=voice`. If it prints `voice_failed:<reason>` it already fell back to text: show that as the design working.

## 31–33 · Metrics

```bash
uv run mira run-all --llm fake
```
7/7 pass, 5 autonomous closes, **false closes 0**, escalation precision/recall 1.00, invariant violations 0, PHI leaks 0. Be explicit: this proves orchestration and guards, not extraction accuracy.

## Likely questions

**Why not one autonomous agent with tools?** The commitments are where trust lives: one invented date or a supplier that doesn't take assignment is a bill for Eleanor. An agent loop makes that a probability; a reducer makes it an invariant I can assert every step. The agents still do all the language work.

**Why not LangGraph?** No LLM decides control flow, so a graph framework adds nothing. Transitions are pure functions over an event log; for durability I'd reach for Temporal, which fits a pure planner directly.

**How do you evaluate?** Two layers. Orchestration: named scenarios with hidden fact sheets, hard gate false closes = 0, escalation precision/recall, invariants, PHI leaks. AI components: a labelled utterance set scored per field (extraction), groundedness and handoff recall (companion), injection success. Today the first layer runs offline; the second is the 1-day item.

**Cost per case?** The ledger records tokens per call and the run summary prints USD for known models. Shape: happy path is 8 calls, ghost_failover 23; each call is ~10–15 turns plus two extraction passes. I'll quote the number from a live run, not a guess. Big lever: supplier memory turns screens into re-verifications.

**What breaks at 100×?** One process, one SQLite file per case, serial calls; no shared supplier view, so three cases cold-call Lakeshore three times. Fix order: per-supplier batching and memory, then Temporal workers and a shared Postgres log. The pure core does not change.

**HIPAA / PHI?** Minimum necessary by construction: screening calls carry no PHI; name/DOB enter a call only after code marks the supplier qualified *and* the org is verified on the call; MBI is never spoken; an injection attempt locks PHI. Production needs a BAA with the model and voice vendors, encryption at rest, access audit and retention policy — not built.

**What if voice fails live?** It falls back to text per call automatically (`text_fallback`); the case continues. If LiveKit is down entirely: `mira call pcp --text --llm live`.

**What did you verify vs assume?** Offline numbers are real and reproducible. Live LLM quality and voice need the live eval. Medicare cost figures ($283 deductible, 13-month capped rental) and K0001's face-to-face status are flagged to verify.

## Fallback plan

| Failure | Do this |
|---|---|
| OpenAI key / model error | Every command with `--llm fake` (default). Same core, same guards, deterministic output |
| LiveKit / audio | `uv run mira call pcp --text --llm live`, or `--llm fake` if the LLM is also down |
| `chat --llm live` slow | `mira chat <case_id>` (fake keyword companion); safety net + grounding still demonstrable |
| Terminal too narrow | `export COLUMNS=160`, rerun |
| Anything else | `uv run pytest -q` (320 passed, 1 skipped) and walk the timeline of a stored case with `mira timeline <case_id>` |
