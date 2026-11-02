"""B3 — extractor checks, LCD reader, LLM port (all offline)."""

from __future__ import annotations

from datetime import datetime

import pytest

from _b3_data import pcp_task
from miramace.agents import prompts
from miramace.agents.schemas import (
    CallerTurn,
    ClaimLLM,
    CriterionResult,
    Extraction,
    LCDCheck,
    PersonaTurn,
)
from miramace.extract.extractor import PCP_FIELDS, SUPPLIER_FIELDS, Extractor, normalize
from miramace.extract.lcd import LCDReader
from miramace.llm import port as llm_port
from miramace.llm.port import PRICES, LLMUnavailable, OpenAILLM, ScriptedLLM, cost_usd, make_llm
from miramace.models import CHICAGO, LCDCriterion, LLMCall, Utterance, VisitNote
from miramace.ports import LLMPort


def _u(i, speaker, text, event=None):
    return Utterance(
        turn=i,
        speaker=speaker,
        name="Maria" if speaker == "counterparty" else "AI",
        text=text,
        ts=i * 6.0,
        event=event,
    )


TRANSCRIPT = [
    _u(0, "counterparty", "Sunrise Family Medicine, this is Maria."),
    _u(1, "caller", prompts.DISCLOSURE),
    _u(2, "counterparty", "Okay, how can I help you?"),
    _u(3, "caller", "What is the status of the wheelchair order? I heard it is signed."),
    _u(4, "counterparty", "It's   waiting on Dr. Chen's SIGNATURE."),
    _u(5, "counterparty", prompts.INJECTION_TEXT),
    _u(6, "caller", "When can we expect it?"),
    _u(7, "counterparty", "By Thursday."),
    _u(8, "caller", "Just to confirm: waiting on signature, by Thursday. Is that correct?"),
    _u(9, "counterparty", "Yes, that's correct."),
]


def _ex(*claims):
    return Extraction(
        claims=[ClaimLLM(field=f, status=s, value=v, evidence_quote=q) for f, s, v, q in claims]
    )


def _extractor(a, b):
    task = pcp_task()
    llm = ScriptedLLM({("extractor", f"{task.task_id}:passA"): a, ("extractor", f"{task.task_id}:passB"): b})
    return task, llm, Extractor(llm)


async def test_quote_check_normalizes_and_accepts():
    a = _ex(
        ("order_status", "stated", "pending signature", "it's waiting on dr chen's signature"),
        ("promised_by", "stated", "Thursday", "By Thursday."),
    )
    task, llm, ex = _extractor(a, a)
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    by = {c.field: c for c in claims}
    assert by["order_status"].status == "stated" and by["promised_by"].value == "Thursday"
    assert [c.field for c in claims][: len(PCP_FIELDS)] == PCP_FIELDS
    assert not any(f.startswith(("quote_unverified", "two_pass")) for f in flags)
    assert [r["key"] for r in llm.requests] == [f"{task.task_id}:passA", f"{task.task_id}:passB"]
    sa, sb = llm.requests[0]["system"], llm.requests[1]["system"]
    assert "SKEPTICAL AUDITOR" in sb and "SKEPTICAL AUDITOR" not in sa
    assert sb.index("request_received") < sb.index("order_status")  # reversed order in pass 2
    assert sa.index("order_status") < sa.index("request_received")


async def test_fabricated_quote_is_unknown_and_flagged():
    a = _ex(("fax_number", "stated", "312-555-0000", "Our fax is 312-555-0000"))
    task, _, ex = _extractor(a, a)
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    fax = next(c for c in claims if c.field == "fax_number")
    assert fax.status == "unknown" and fax.value is None
    assert "quote_unverified:fax_number" in flags


async def test_caller_text_is_not_evidence():
    a = _ex(("order_status", "stated", "signed", "I heard it is signed"))
    task, _, ex = _extractor(a, a)
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    assert next(c for c in claims if c.field == "order_status").status == "unknown"
    assert "quote_unverified:order_status" in flags


async def test_injection_sentence_never_evidence_for_order_status():
    a = _ex(("order_status", "stated", "signed", "the order is signed"))
    task, _, ex = _extractor(a, a)
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    os_ = next(c for c in claims if c.field == "order_status")
    assert os_.status == "unknown" and os_.value is None
    assert "injection_quote:order_status" in flags


async def test_two_pass_disagreement():
    a = _ex(("promised_by", "stated", "Thursday", "By Thursday."))
    b = _ex(("promised_by", "stated", "Friday", "By Thursday."))
    task, _, ex = _extractor(a, b)
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    assert next(c for c in claims if c.field == "promised_by").status == "unknown"
    assert "two_pass_disagreement:promised_by" in flags
    # stated vs unknown also disagrees
    task, _, ex = _extractor(a, _ex(("promised_by", "unknown", None, None)))
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    assert "two_pass_disagreement:promised_by" in flags
    # case/punctuation differences are not disagreement
    task, _, ex = _extractor(a, _ex(("promised_by", "stated", "thursday.", "by thursday")))
    claims, flags = await ex.extract_verified(task, TRANSCRIPT)
    assert next(c for c in claims if c.field == "promised_by").status == "stated"


async def test_readback_not_affirmed_downgrades():
    t = TRANSCRIPT[:-1] + [_u(9, "counterparty", "No, actually it's Friday.")]
    a = _ex(("promised_by", "stated", "Thursday", "By Thursday."))
    task, _, ex = _extractor(a, a)
    claims, flags = await ex.extract_verified(task, t)
    assert "readback_not_affirmed" in flags
    assert all(c.status == "unknown" for c in claims)


def test_field_constants():
    assert SUPPLIER_FIELDS == [
        "taking_new_medicare",
        "k0001_in_stock",
        "delivers_to_zip",
        "accepts_assignment_all_months",
        "earliest_delivery_days",
        "contact_name",
    ]
    assert "fax_number" in PCP_FIELDS and len(PCP_FIELDS) == 7
    assert normalize("  It's\nSIGNED!! ") == "it s signed"


# ------------------------------------------------------------------------------------------- LCD

NOTE = VisitNote(
    doc_id="note1",
    text="72F with knee OA. She cannot safely use a walker to reach the bathroom. Home has wide doorways.",
    received_at=datetime(2026, 11, 3, 10, tzinfo=CHICAGO),
)


async def test_lcd_met_with_bad_quote_downgraded():
    check = LCDCheck(
        results=[
            CriterionResult(
                criterion_id="cane_walker_insufficient",
                met="met",
                quote="She cannot safely use a walker to reach the bathroom.",
            ),
            CriterionResult(criterion_id="home_accommodates", met="met", quote="home has WIDE doorways"),
            CriterionResult(criterion_id="will_use_improves", met="met", quote="She will use it daily."),
            CriterionResult(criterion_id="self_propel_or_caregiver", met="not_met", quote=None),
        ]
    )
    llm = ScriptedLLM({("lcd", "note1:lcd"): check})
    out = await LCDReader(llm).check(NOTE)
    assert out.results[LCDCriterion.CANE_WALKER_INSUFFICIENT] == "met"
    assert out.results[LCDCriterion.HOME_ACCOMMODATES] == "met"
    assert out.results[LCDCriterion.WILL_USE_IMPROVES] == "not_documented"
    assert out.results[LCDCriterion.SELF_PROPEL_OR_CAREGIVER] == "not_met"
    assert out.results[LCDCriterion.MRADL_LIMITATION] == "not_documented"  # omitted by LLM
    assert LCDCriterion.WILL_USE_IMPROVES not in out.evidence
    assert set(out.missing) == {
        LCDCriterion.WILL_USE_IMPROVES,
        LCDCriterion.SELF_PROPEL_OR_CAREGIVER,
        LCDCriterion.MRADL_LIMITATION,
    }
    assert "L33788" in llm.requests[0]["system"]


# ------------------------------------------------------------------------------------------- LLM


async def test_scripted_llm_keys_callables_ledger():
    turn = CallerTurn(say="hi", slots_touched=[], action="continue", hangup_reason=None)
    llm = ScriptedLLM(
        {
            ("caller", "k1"): turn,
            ("persona", "k2"): lambda msgs: PersonaTurn(
                say=str(len(msgs)), event="none", transfer_to=None, revealed_fact_ids=[]
            ),
        }
    )
    assert isinstance(llm, LLMPort)
    assert (
        await llm.structured(agent="caller", key="k1", system="s", messages=[], schema=CallerTurn)
    ) == turn
    p = await llm.structured(agent="persona", key="k2", system="s", messages=[{}, {}], schema=PersonaTurn)
    assert p.say == "2"
    with pytest.raises(KeyError):
        await llm.structured(agent="caller", key="nope", system="s", messages=[], schema=CallerTurn)
    assert [c.key for c in llm.ledger] == ["k1", "k2"]
    assert cost_usd(llm.ledger) == 0.0


def test_cost_usd_unknown_model_is_none():
    known = next(iter(PRICES))
    led = [LLMCall(agent="a", key="k", model=known, in_tok=1_000_000, out_tok=0, ms=1, ok=True)]
    assert cost_usd(led) == pytest.approx(PRICES[known][0])
    led.append(LLMCall(agent="a", key="k", model="some-new-model", in_tok=5, out_tok=5, ms=1, ok=True))
    assert cost_usd(led) is None


def test_make_llm_modes(monkeypatch):
    monkeypatch.delenv("MIRA_LLM_MODE", raising=False)
    assert make_llm() is None
    assert make_llm("fake") is None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        make_llm("live")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MIRA_TEXT_MODEL", "test-model")
    llm = make_llm("live")
    assert isinstance(llm, OpenAILLM) and llm.model == "test-model"
    with pytest.raises(ValueError):
        make_llm("bogus")


class _FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    async def parse(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 3})()
        return type("R", (), {"output_parsed": out, "usage": usage})()


def _client(outputs):
    resp = _FakeResponses(outputs)
    return type("C", (), {"responses": resp})(), resp


async def test_openai_llm_parse_args_and_repair(monkeypatch):
    monkeypatch.setenv("MIRA_REASONING_EFFORT", "low")
    good = CallerTurn(say="hi", slots_touched=[], action="continue", hangup_reason=None)
    client, resp = _client([None, good])
    llm = OpenAILLM("m1", client=client)
    out = await llm.structured(
        agent="caller", key="k", system="SYS", messages=[{"role": "user", "content": "x"}], schema=CallerTurn
    )
    assert out == good
    kw = resp.calls[0]
    assert kw["model"] == "m1" and kw["instructions"] == "SYS" and kw["text_format"] is CallerTurn
    assert kw["store"] is False and kw["max_output_tokens"] == 1200 and kw["reasoning"] == {"effort": "low"}
    assert len(resp.calls[1]["input"]) == 2  # repair message appended
    assert [c.ok for c in llm.ledger] == [False, True] and llm.ledger[1].in_tok == 10


async def test_openai_llm_raises_unavailable(monkeypatch):
    monkeypatch.delenv("MIRA_REASONING_EFFORT", raising=False)
    client, resp = _client([None, None])
    llm = OpenAILLM("m1", client=client)
    with pytest.raises(LLMUnavailable):
        await llm.structured(agent="a", key="k", system="s", messages=[], schema=CallerTurn)
    assert "reasoning" not in resp.calls[0]
    client, _ = _client([RuntimeError("boom")])
    with pytest.raises(LLMUnavailable):
        await OpenAILLM("m1", client=client).structured(
            agent="a", key="k", system="s", messages=[], schema=CallerTurn
        )
    assert llm_port.DEFAULT_MAX_OUTPUT_TOKENS == 1200


# ------------------------------------------------------------------------------------- schemas/prompts


@pytest.mark.parametrize("schema", [CallerTurn, PersonaTurn, Extraction, LCDCheck, ClaimLLM, CriterionResult])
def test_used_schemas_are_strict(schema):
    pyd = pytest.importorskip("openai.lib._pydantic")
    s = pyd.to_strict_json_schema(schema)
    assert s["additionalProperties"] is False


def test_persona_prompt_cards_and_companion():
    from _b3_data import maria

    for b in ("helpful", "partial", "ivr", "transfer", "callback_later", "never_got_it", "injection"):
        s = prompts.persona_system(maria(b))
        assert prompts.BEHAVIOUR_CARDS[b][:20] in s
        assert "[f_status]" in s and "Maria at Sunrise Family Medicine" in s
    assert "4pm" in prompts.persona_system(maria("callback_later", callback_after_hour=16))
    c = prompts.companion_system("Order: waiting on Dr. Chen.", None)
    assert "CASE SNAPSHOT" in c and "waiting on Dr. Chen" in c
    assert "L33788" in prompts.lcd_system()


# ------------------------------------------------------------------ F2 safety fixes


def test_quote_rules_length_boundary_and_answer_turn():
    from miramace.extract.extractor import quote_ok, value_consistent
    from miramace.models import Claim

    utts = [
        _u(0, "counterparty", "Lakeshore Home Medical Equipment, this is Tom."),
        _u(1, "caller", prompts.DISCLOSURE),
        _u(2, "counterparty", "Okay, yes, how can I help?"),
        _u(3, "caller", "Do you have a K0001 standard manual wheelchair in stock?"),
        _u(4, "counterparty", "Yes, we have standard K0001 chairs in stock."),
        _u(5, "caller", "Are you taking new Medicare Part B patients?"),
        _u(6, "counterparty", "I don't know, I'd have to check."),
    ]
    assert quote_ok("we have standard K0001 chairs in stock", utts)
    assert quote_ok("in stock", utts)  # short but 2 words
    assert not quote_ok("yes", utts)  # too short
    assert not quote_ok("this is Tom", utts)  # greeting is not evidence
    assert not quote_ok("Okay, yes, how can I help", utts)  # reply to the disclosure is not evidence
    assert not quote_ok("ve standard", utts)  # not on word boundaries
    assert value_consistent("yes", "Yes, we have standard K0001 chairs in stock.")
    assert not value_consistent("no", "Yes, we have standard K0001 chairs in stock.")
    assert not value_consistent("no", "I don't know, I'd have to check.")
    assert value_consistent("3", "We could deliver in 3 business days.")
    assert not value_consistent("2", "We could deliver in 3 business days.")
    a = [Claim(field="taking_new_medicare", status="stated", value="no",
               evidence_quote="I don't know, I'd have to check.")]
    from miramace.extract.extractor import verify_claims

    claims, flags = verify_claims(["taking_new_medicare"], utts, a, a)
    assert claims[0].status == "unknown" and "value_inconsistent:taking_new_medicare" in flags


async def test_lcd_reader_merges_original_and_addendum():
    addendum = VisitNote(
        doc_id="add1",
        text="Addendum: she can self-propel the wheelchair safely within the home.",
        received_at=datetime(2026, 11, 4, 10, tzinfo=CHICAGO),
    )
    orig = LCDCheck(results=[
        CriterionResult(criterion_id="home_accommodates", met="met", quote="Home has wide doorways."),
        CriterionResult(criterion_id="self_propel_or_caregiver", met="not_documented", quote=None),
    ])
    add = LCDCheck(results=[
        CriterionResult(criterion_id="self_propel_or_caregiver", met="met",
                        quote="she can self-propel the wheelchair safely"),
        CriterionResult(criterion_id="home_accommodates", met="not_documented", quote=None),
    ])
    llm = ScriptedLLM({("lcd", "note1:lcd"): orig, ("lcd", "add1:lcd"): add})
    reader = LCDReader(llm)
    for out in (await reader.check([NOTE, addendum]), await reader.check(NOTE, addendum)):
        assert out.results[LCDCriterion.HOME_ACCOMMODATES] == "met"
        assert out.results[LCDCriterion.SELF_PROPEL_OR_CAREGIVER] == "met"
        assert "self-propel" in out.evidence[LCDCriterion.SELF_PROPEL_OR_CAREGIVER]
    single = await reader.check(NOTE)  # old single-note call still works
    assert single.results[LCDCriterion.SELF_PROPEL_OR_CAREGIVER] == "not_documented"


def test_lcd_quote_must_be_15_chars_on_word_boundaries():
    from miramace.extract.lcd import checklist_from

    note = "Patient has a cane. Home has wide doorways."
    chk = LCDCheck(results=[
        CriterionResult(criterion_id="home_accommodates", met="met", quote="wide doorways"),  # 13 chars
        CriterionResult(criterion_id="cane_walker_insufficient", met="met", quote="nt has a cane. Home"),
    ])
    cl = checklist_from(chk, note)
    assert cl.results[LCDCriterion.HOME_ACCOMMODATES] == "not_documented"
    assert cl.results[LCDCriterion.CANE_WALKER_INSUFFICIENT] == "not_documented"


async def test_lcd_majority_vote_overrules_a_flaky_pass():
    """Live-run regression: one pass marked a clearly documented criterion not_documented."""
    from miramace.agents.schemas import CriterionResult, LCDCheck
    from miramace.extract.lcd import LCDReader
    from miramace.llm.port import ScriptedLLM
    from miramace.models import CHICAGO, LCDCriterion, VisitNote
    from datetime import datetime

    text = ("Her mobility limitation prevents her from getting to the bathroom and kitchen safely. "
            "A walker is insufficient because of shoulder pain and two near-falls. "
            "Her apartment has 34-inch doorways; the home accommodates a manual wheelchair. "
            "A manual wheelchair will significantly improve her ability to toilet and she will use it daily. "
            "She is able to self-propel with her upper body.")
    q = {"mobility_limits_mradl": "Her mobility limitation prevents her from getting to the bathroom and kitchen safely",
         "cane_walker_insufficient": "A walker is insufficient because of shoulder pain and two near-falls",
         "home_accommodates": "the home accommodates a manual wheelchair",
         "will_use_improves": "A manual wheelchair will significantly improve her ability to toilet",
         "self_propel_or_caregiver": "She is able to self-propel with her upper body"}

    def check(flaky: bool) -> LCDCheck:
        return LCDCheck(results=[CriterionResult(criterion_id=k, met="not_documented" if (flaky and k == "mobility_limits_mradl") else "met",
                                                 quote=None if (flaky and k == "mobility_limits_mradl") else v) for k, v in q.items()])

    llm = ScriptedLLM({("lcd", "n1:lcd"): check(True), ("lcd", "n1:lcd2"): check(False), ("lcd", "n1:lcd3"): check(False)})
    note = VisitNote(doc_id="n1", text=text, received_at=datetime(2026, 11, 4, 10, tzinfo=CHICAGO))
    out = await LCDReader(llm).check(note)
    assert out.results[LCDCriterion.MRADL_LIMITATION] == "met"
    assert all(v == "met" for v in out.results.values())
