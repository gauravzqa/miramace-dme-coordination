# Writeup — DME coordination prototype

Gaurav · Mira Mace take-home · Senior AI Engineer, Agents

## Sequencing

I started from where a claim actually dies. Medicare pays for Eleanor's chair only if there is a **written order** (a complete Standard Written Order) *and* a visit note showing **medical necessity under the LCD** (mobility limit in the home, cane/walker not enough, the home fits the chair, she can use it). A slow supplier costs days; a bad order costs the claim. So the **PCP order leg is the critical path and the main claim risk**, and I built it deep: request → visit-note LCD check → addendum naming the missing criterion → SWO completeness and K0001 check → order faxed *to the committed supplier* → receipt confirmed.

The **supplier leg is broad and thin**: rank by distance, screen with no PHI, commit one, keep a backup, re-verify within 24h of scheduling. Two Medicare details shaped it: the supplier must **accept assignment for every rental month** (a K0001 is capped rental, so one "no" in month 7 is Eleanor's bill), and the order must go to the supplier who will actually bill.

The **patient companion** came third. It is the voice of the case, not its owner, and it may only say what the snapshot says. Cost messaging is generated from code: price is set by Medicare, not quoted; the 2026 Part B deductible ($283), then 20% of the approved monthly rental for up to 13 months, then she owns it. *These cost figures need verifying against the current fee schedule before a real patient hears them.*

Each failure mode from the brief became a **named scenario** with a hidden fact sheet and an expected outcome (7 total), so "does it handle a ghosting supplier?" is a command.

I built the **event-sourced deterministic core first**, before any prompt. If the core is right, every LLM component is replaceable and testable behind a port; if the core is wrong, no prompt saves it.

## Technology & architecture

Python 3.12, Pydantic v2, plain `asyncio`, `typer` + `rich`. Each case is an **append-only SQLite event log**; a **pure reducer** projects it to `CaseState`; a **pure planner** maps `(state, now)` to the desired tasks; tasks go through an **idempotent outbox** keyed `(case, party, intent, version)`; everything runs on a **virtual clock** with business hours, so a 15-day case replays in seconds. Five invariants are checked after every step (e.g. `SCHEDULED` ⇒ order received by the committed supplier ∧ valid K0001 ∧ assignment confirmed ∧ verified within 24h ∧ cost accepted).

LLMs are called through the OpenAI Responses API with **strict structured outputs** (every schema is a Pydantic model). The PCP leg also runs as **real voice**: two LiveKit agent sessions (caller ↔ front-desk persona) on a local server, OpenAI STT/TTS, sharing the same "brains" and guards as the text transport, with automatic fallback to text.

```
 timers · call results · inbound faxes · patient intents · human resolutions
                              │ events (each cites a source)
                              ▼
 event log ──reduce──▶ CaseState ──plan──▶ tasks (outbox, idem_key) ──▶ executor
     ▲                    (pure)    (pure)                               │
     └────── interpret: claims → events (rules decide) ◀── calls / docs / companion (LLM)
```

**Why not LangGraph:** there is no LLM-driven control flow to orchestrate; transitions are pure functions. **Why not Temporal yet:** the log plus outbox already gives replay and idempotency for one process; because the planner is pure, Temporal becomes a drop-in durable runtime when there are many cases and real timers.

## Where AI sits — and where I kept it out

| AI in (language is unavoidable) | Code only (deterministic, auditable) |
|---|---|
| Conducting calls with the PCP desk and suppliers | Every state transition, owner, due date, retry, failover |
| Extracting claims from transcripts and visit notes | Supplier eligibility, SWO completeness, LCD pass/fail, K0001 drift, assignment |
| Talking with Eleanor: explain, reassure, detect intent | Which supplier to commit; when to contact Eleanor; every date/$/name she hears |
| Persona simulation (test harness only) | When PHI may enter a call; which escalation opens |

The guards that make this hold:
- **PHI gate + injection lock.** Supplier screening calls carry no PHI. Name/DOB enter a `CallTask` only after code marks the supplier qualified, and are spoken only after the org is verified on the call. The Medicare number is never spoken. A counterparty line that looks like a prompt injection locks PHI for the rest of the call and can never be used as evidence.
- **Evidence quotes, two-pass, readback.** Every claim needs a verbatim quote that must be a substring of one counterparty line; the value must agree with the quote; two extraction passes must agree; a denied readback turns the field to `unknown`. No confidence scores, just `stated` or `unknown`.
- **Grounding + safety net.** Every companion message is checked: each date, amount and name must exist in the snapshot, else regenerate once, then a safe template. A keyword net turns distress or "a person, please" into `needs_human` even if the model misses it.
- **The LLM never mutates state.** No agent has a tool that writes state. Callers return transcripts, the extractor returns claims, the companion returns typed intents; code validates and appends events.

## The cut list

- **Real telephony** (PSTN, IVR navigation, voicemail detection). Voice runs over local LiveKit on the PCP leg only; supplier and patient legs are text over the same port.
- **Claim submission, PCP portal automation, fax gateway.** Simulated inbound documents.
- **Geocoding.** ZIP-centroid distance; the supplier's "we deliver to 60614" answer is authoritative.
- **Multi-case scheduling, auth, HIPAA controls** (BAA, audit access, encryption at rest). Designed for, not built.
- **Long-term companion memory** beyond the patient profile.
- **Live eval set and replay cassettes.** Fake mode proves orchestration; it does not prove extraction or tone.

## What's next

**1 day:** real voice on the supplier legs and the companion; a hand-labelled live eval set (~50 nasty utterances: hedges, contradictions, IVRs, social engineering for DOB/MBI) scored per field; recorded cassettes so live runs replay deterministically in CI; measured cost and latency per call.

**2 weeks:** (1) **shadow mode** beside advocates on real cases, scoring agreement and advocate minutes saved; (2) **supplier memory** across cases (stock, assignment, responsiveness, ghosting history); (3) a **multi-case scheduler** with per-supplier batching ("three patients need K0001 from Lakeshore"); (4) **Temporal** for durable timers. **Why this order:** measure before scaling, because I don't know yet which failure modes dominate in reality; memory makes scaling cheap, because most supplier calls become re-verifications instead of cold screens; only then is throughput the bottleneck.

**How I'd measure it:** days from verbal order to delivery; advocate minutes per case; autonomous closes, gated on **false closes = 0**.

**Verified vs not.** Offline, with fakes behind the same ports: all 7 scenarios meet expectations, 5 autonomous closes, false closes 0, escalation precision/recall 1.0, invariant violations 0, PHI before qualification 0; 356 tests pass (including live-wiring tests against a stand-in model). Live LLM mode and the LiveKit call are wired and unit-tested against stubs, but none of the numbers above come from them; they need a live smoke run (`docs/voice-smoke.md`). Extraction accuracy, companion tone and cost per case still need the live eval set before I'd trust any number.
