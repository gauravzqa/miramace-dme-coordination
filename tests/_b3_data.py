"""Shared offline fixtures for B3 tests (PCP task + Maria persona, supplier task)."""

from __future__ import annotations

from miramace.models import CallTask, Fact, PersonaSim, PersonaSpec, SlotSpec, eleanor

ORG = "Sunrise Family Medicine"

PCP_SLOTS = [
    SlotSpec(
        id="request_received",
        ask="Did you receive our order request for her wheelchair?",
        type="bool",
        required=True,
    ),
    SlotSpec(id="order_status", ask="What is the status of the wheelchair order?", type="str", required=True),
    SlotSpec(id="promised_action", ask="What will the office send us?", type="str", required=True),
    SlotSpec(id="promised_by", ask="When can we expect it?", type="str", required=True),
    SlotSpec(id="named_owner", ask="Who is handling it?", type="str", required=True),
    SlotSpec(id="fax_number", ask="What fax number should we use?", type="str", required=False),
]

MARIA_FACTS = [
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


def pcp_task(phi: bool = True, **kw) -> CallTask:
    return CallTask(
        task_id=kw.pop("task_id", "case1:pcp:request_order:v1"),
        party="pcp",
        party_id="pcp",
        org_name=ORG,
        phone="(312) 555-0198",
        intent="request_order",
        purpose="Request the visit note and a signed standard written order for a K0001 manual wheelchair.",
        slots=PCP_SLOTS,
        phi=eleanor().phi() if phi else None,
        context={"visit": "a patient of Dr. Chen seen 3 days ago"},
        **kw,
    )


def maria(behaviour: str = "helpful", **sim) -> PersonaSpec:
    return PersonaSpec(
        party_id="pcp",
        role="pcp",
        org_name=ORG,
        answerer_name="Maria",
        behaviour=behaviour,
        facts=MARIA_FACTS,
        sim=PersonaSim(**sim),
    )
