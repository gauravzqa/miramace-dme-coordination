# Scenario YAML schema

Loaded by `miramace.scenario.load_scenario(name_or_path)` (names resolve to `scenarios/<name>.yaml`) and validated
by `scenario.Scenario` (unknown keys are rejected).

```yaml
name: happy_path                      # must equal the file stem
description: one line
patient:
  address: "2150 N Lincoln Park W, Chicago, IL 60614"   # informational; the patient confirms it via the script
  best_hours: [9, 20]
  caregiver_name: Ana                 # optional: caregiver_phone, home_access, hearing_or_language, prior_concerns
patient_script:                       # fakes/patient.py ScriptedPatient, keyed by NotifyPurpose
  welcome_intake: [{kind: confirm_address, address: "..."}, {kind: give_consent}]
  cost_disclosure: [{kind: cost_accepted}]          # or [{kind: cost_concern}] -> COST_CONCERN escalation
  delivery_slot: [{kind: confirm_delivery_slot}]
  delivered_checkin: [{kind: delivery_confirmed}]
personas:                             # party_id -> models.PersonaSpec; party_id is filled from the key
  pcp: {role: pcp, org_name: Sunrise Family Medicine, answerer_name: Maria, behaviour: helpful,
        facts: [...], sim: {...}}
  <supplier sid>: {role: supplier, org_name: <exact name from data/suppliers.csv>, ...}
ground_truth: {claim_would_pass: true, lcd_supported: true, hcpcs: K0001}
expect: {final: DONE, escalations: [], max_sim_days: 14}   # final: DONE|BLOCKED_ON_HUMAN|STUCK|MAX_STEPS|CANCELLED
```

Validation: `org_name` must be `Sunrise Family Medicine` for `pcp` and the exact directory name for a supplier sid
(the caller verifies the party by that name). Suppliers without a persona never answer.

## Facts: slot ids per call (fake persona answers by slot id; extractor value = `fact.value`)

| Call (intent) | Slots (required first) | Value format |
|---|---|---|
| pcp `request_order` | `promised_action`, `promised_by`, (`named_owner`) | promised_by: "Thursday", "by Friday", "tomorrow", "in 3 days" |
| pcp `followup_order_k` | `request_received`, (`order_status`, `promised_by`, `named_owner`) | yes / no |
| pcp `request_addendum` | `will_send_addendum`, `promised_by` | yes / no |
| pcp `request_correction` | `will_send_correction`, `promised_by` | yes / no |
| pcp `send_order_<sid>` | `will_fax_order`, (`promised_by`) | yes / no |
| supplier `screen` (no PHI) | `taking_new_medicare`, `k0001_in_stock`, `delivers_to_zip`, `accepts_assignment_all_months`, (`earliest_delivery_days`, `contact_name`) | yes / no, int |
| supplier `commit` | `reserve_unit`, `accepts_assignment_all_months`, (`reference_number`) | yes / no |
| supplier `verify_order_received` | `order_received` | yes / no |
| supplier `reverify` | `k0001_in_stock`, `accepts_assignment_all_months` | yes / no |
| supplier `propose_slot` | `delivery_slot` | "tomorrow at 10 AM", "in 2 days at 1 PM", "Thursday 10am" |
| supplier `verify_delivered` | `delivered` | yes / no |

Engine slots a persona does not define get cooperative defaults (`fakes/sim.py DEFAULT_FACTS`), e.g. a supplier
without `reserve_unit` will hold a unit, `delivery_slot` defaults to "tomorrow 10am". Define a fact to override.
Fake callers ask only the required slots.

## `sim` (models.PersonaSim)

| Field | Effect |
|---|---|
| `no_answer_attempts: N` | first N calls to this party are NO_ANSWER |
| `ghost_after: never\|commit\|order_sent` | supplier stops answering (and never delivers) after commit / after the order is faxed to it |
| `lost_first_request: true` | pcp: request v1 is lost (no docs); the follow-up hears "We never received any request for her." -> resend v2 |
| `refuses: true` | pcp: request_order is REFUSED -> PCP_REFUSED escalation |
| `note_delay_bh`, `signed_delay_bh` | business hours after ORDER_REQUESTED until the visit note / signed SWO fax arrives |
| `fax_delay_bh` | business hours until a corrected SWO arrives after CORRECTION_REQUESTED |
| `visit_note_text`, `note_claims` | the note fax and its LCD claims (`field`: LCD criterion, `evidence_quote` must be verbatim in the text; `status` defaults to stated, `value` to met) |
| `addendum_text`, `addendum_claims` | addendum fax after ADDENDUM_REQUESTED (arrives `note_delay_bh` later); LCD re-evaluated cumulatively |
| `swo_signed` | dict of `models.SWO` fields (the signed order fax). Optional key `corrected: {...}` = overrides used for the corrected SWO; without it the requested fields are filled with complete values |
| `callback_after_hour`, `contradicts_on_readback` | handled by the persona brain (behaviour `callback_later`, readback denial) |

Delivery: after the patient confirms the slot (DELIVERY_SCHEDULED), a proof-of-delivery fax arrives at the slot time
unless the supplier ghosts.
