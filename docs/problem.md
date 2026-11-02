# Take-Home: DME Back-End Coordination (Mira Mace)

> Source: [Google Doc](https://docs.google.com/document/d/10ZdP5EmCPSHYR_6s8XfPCiTS_GWE5G5PnC-YOGEjxQU/edit) sent by Abhinav Garg (abhinav@miramace.com) on 2026-10-27.
> **Format:** ~3-hour time-boxed exercise. Build a working slice of an AI workflow for their care navigation work.
> **Presentation:** Tue Nov 3, 2026, 2:30–3:15am IST (Google Meet: https://meet.google.com/gsm-twap-fsq). 45 min live: run the demo, answer questions, defend the design.
> **Before the session:** send the GitHub link so they can review it async.

---

## The problem, in our words

Many of our patients come to us because navigating Medicare durable medical equipment (DME) is a mess. They need a wheelchair, CPAP, hospital bed, walker, or oxygen concentrator. It's a coordination problem across three parties: the patient, their primary care doctor (PCP, who writes the order), and the DME supplier (who delivers the equipment and bills insurance). Every handoff is friction, and the patient ends up chasing all three. We exist to remove that friction.

Today, care advocates on our team do this work end-to-end. Intake is handled: a patient has already called in and we have their situation documented. What follows is the hard part: coordinating across supplier, PCP, and Medicare, without a care advocate chasing each party by hand.

We want to automate that coordination. The north star is no care advocate in the loop at all. A system that runs the case itself and comes back to a human only when it genuinely has to. We're nowhere near that today. That's part of what we want you to figure out.

## Background: the mechanics

**Medicare** is US government health insurance, mainly for people 65 and older. For equipment like a wheelchair, Medicare won't pay unless two things are true:

1. The patient's doctor has written a formal **order** (a prescription) stating the equipment is medically necessary.
2. The **supplier** is *enrolled* with Medicare.

Miss either one and the claim is denied. The patient gets stuck with the full bill, or simply never gets the equipment. The supplier delivers the wheelchair and then bills Medicare directly; the patient typically owes a small share (~20%). Nobody gets paid until the order, the supplier's enrollment, and the billing codes all line up.

That's why this is a coordination problem: the doctor, the supplier, and Medicare each hold one piece, and the patient is left chasing all three. US healthcare still runs heavily on **voice**, which is why "contacting the PCP" or "contacting a supplier" means a phone call, not a clean API.

## What the care advocates do today

Right now a care advocate runs each case by hand, start to finish. For a case like Eleanor's, that's roughly:

1. **Work the supplier list.** Call enrolled suppliers from the directory. Ask each: taking new Medicare patients? Stock a standard manual wheelchair (K0001)? How soon could you deliver? Note who picks up, who doesn't, and call the no-answers back later.
2. **Chase the written order.** Contact the PCP's office for the formal order: call the front desk or work their portal, confirm it's moving, nudge again days later if it's still not signed.
3. **Match and hand off.** Once there's a real order and a supplier that's enrolled, in stock, and responsive, connect the two and get delivery scheduled, making sure the billing code on the claim matches what was ordered.
4. **Keep the patient in the loop.** Call along the way: what's happening, what she needs to do, what she'll owe (~20%), when to expect the equipment.

One care advocate carries multiple of these at once.

## Where it goes wrong

The steps are simple. The failure modes are where the time goes:

- **The supplier can't actually serve her.** Enrolled but not taking new patients, out of stock on K0001, or outside her area. Back to the list.
- **A supplier says yes, then goes silent.** Agrees to deliver, then never calls back and never delivers. The care advocate finds out only by chasing. Days lost.
- **The request falls in a hole.** The order request never reaches the right person, or lands in a queue no one works. The office says "we never got it." The order stalls.
- **The order is wrong.** Wrong billing code or equipment description. The claim gets rejected downstream and the case restarts.
- **The patient goes quiet.** Can't be reached for follow-up, or is confused about what she owes and disengages.

Any one of these can stall a case for a week. A care advocate's real skill is catching them early and working around them.

## The case you're starting with

Intake is complete:

```
Patient:    Eleanor Martinez, 72
Medicare:   Original Medicare (Part B), no supplemental plan
Equipment:  Standard manual wheelchair
PCP:        Dr. Sarah Chen, Sunrise Family Medicine, Chicago IL
PCP phone:  (312) 555-0198
Status:     PCP visit completed 3 days ago, verbal order noted in chart,
            written order not yet submitted
Suppliers:  None contacted yet
```

*Medicare Part B covers durable medical equipment. A "supplemental" plan (Medigap) would cover some of the costs Medicare doesn't. Eleanor doesn't have one.*

## What you're working with

- **Medicare supplier directory** (below): names, phone numbers, and addresses of enrolled DME suppliers in the area. That's it. No stock info, no in-network flags, no responsiveness history. Addresses tell you who's plausibly in range; they tell you nothing about whether a supplier will actually serve Eleanor. Sparse on purpose. **Don't invent richer structure; work with what's here.**
- **Medicare's official coverage documentation** for standard manual wheelchairs: what qualifies, what paperwork is required, how the claim gets coded and submitted (billing code K0001). Start here: https://www.medicare.gov/coverage/wheelchairs-scooters
- **No supplier API and no portal.** Reaching a supplier means the phone, one at a time.

## What we're asking for

Build a prototype of that automated system.

Eleanor's case has to get worked with the care advocate asleep. The system gets as far as it can on its own and comes back to a human only when it genuinely has to. Everything else is your call: what it does at each handoff, what triggers the next step, what happens when things go wrong.

There's more than one good answer. Some people will build the whole loop end to end, each part doing a plausible version of its job. Others will take the piece they think matters most and genuinely automate it, sketching enough of the rest that we can still see the whole. Both are good.

What we're reading for is the same either way: something here has to actually do the work, and we should be able to see from it what the automated version of this job looks like.

## Constraints

- **Around 3 hours.** Time-boxing is suggested. The scope is deliberately bigger than the time. Prioritize, and tell us what you cut.
- Skip auth, persistent DB, UI polish. In-memory or a JSON file is fine. **Mock external systems freely, but keep real LLM calls**, and be explicit about what's mocked.
- Any stack, any model, any framework. Use what you'd actually reach for in production.
- Use AI freely while building. We don't care who typed the code. **We care a great deal about where AI sits inside the running system, and where you deliberately kept it out.**

## Deliverable

1. **The code.** A repo or zip we can read and run. Include a short README: how to run the demo.
2. **A short 1–2 page writeup** covering:
   - **Sequencing.** What you built, and how you decided.
   - **Technology & architecture.** What you reached for and why: stack, frameworks, how the pipeline is wired.
   - **The cut list.** What you deliberately *didn't* do, and why.
   - **What's next.** With 1 more day, what do you build? With 2 weeks? Why that order?

## How the process works

Send the code, the writeup, and a way to run the demo. They review async. Then 45 min live: you run the demo, they ask questions, you defend.

---

## Sample supplier directory

| supplier_name | phone | address |
|---|---|---|
| Lakeshore Home Medical Equipment | (312) 555-0142 | 1820 N Clark St, Chicago, IL 60614 |
| Windy City Mobility & DME | (312) 555-0177 | 435 W Erie St, Chicago, IL 60654 |
| Prairie State Medical Supply | (773) 555-0211 | 3344 N Western Ave, Chicago, IL 60618 |
| Northside CarePlus Equipment | (773) 555-0288 | 5215 N Clark St, Chicago, IL 60640 |
| ChicagoLand Durable Medical | (312) 555-0301 | 900 S Wabash Ave, Chicago, IL 60605 |
| Roosevelt Medical Supply Co | (312) 555-0319 | 1147 W Roosevelt Rd, Chicago, IL 60608 |
| Halsted Home Health Supply | (773) 555-0356 | 2870 N Halsted St, Chicago, IL 60657 |
| Midwest Mobility Solutions | (847) 555-0402 | 6601 W Touhy Ave, Niles, IL 60714 |
| Belmont Medical Equipment | (773) 555-0447 | 3400 W Belmont Ave, Chicago, IL 60618 |
| Lincoln Park DME Services | (312) 555-0468 | 2401 N Lincoln Ave, Chicago, IL 60614 |
| South Loop Medical Supply | (312) 555-0490 | 1530 S State St, Chicago, IL 60605 |
| Evanston Home Medical | (847) 555-0533 | 1700 Central St, Evanston, IL 60201 |
