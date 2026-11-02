"""StubResponsesLLM — an offline stand-in for the OpenAI model behind the LIVE wiring.

It implements `LLMPort.structured` and answers every agent the live code paths call with a schema-valid object
produced by the offline rule-based logic:

    caller     -> FakeCallerBrain on the same TranscriptView      (CallerTurn)
    persona    -> FakePersonaBrain on the brain's PersonaSpec     (PersonaTurn)
    extractor  -> one ClaimLLM per field, quoting counterparty answer lines verbatim (Extraction)
    lcd        -> criteria from the scenario's note/addendum claims, quotes copied from the note (LCDCheck)
    companion  -> grounded generic wording for compose; keyword intents for respond (CompanionOut)

Brains only hand the LLM a system prompt + chat messages, so `install(monkeypatch, stub)` wraps
LLMCallerBrain.next / LLMPersonaBrain.next / Extractor.extract_verified to also give the stub the structured
view it needs (the REAL brain code — prompt building, message shaping, guards — still runs).

Every request is validated the way the Responses API would: known agent/schema pairing, non-empty
instructions, `input` = non-empty list of {role, content:str} with a valid role, schema convertible to a
strict
json_schema text format. Problems are collected in `stub.errors` (the engine swallows exceptions, so tests
must assert `stub.errors == []`).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel

from miramace.agents.schemas import (
    CallerTurn,
    ClaimLLM,
    CompanionOut,
    CriterionResult,
    Extraction,
    IntentLLM,
    LCDCheck,
    PersonaTurn,
)
from miramace.calls.brains import FakeCallerBrain, FakePersonaBrain, LLMCallerBrain, LLMPersonaBrain
from miramace.extract.extractor import Extractor, fields_for, normalize
from miramace.llm.port import LLMUnavailable
from miramace.models import LCD_CRITERION_BY_LLM_ID, PCP_PARTY_ID, LCDCriterion, LLMCall
from miramace.rules import _lcd_value

AGENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "caller": CallerTurn,
    "persona": PersonaTurn,
    "extractor": Extraction,
    "lcd": LCDCheck,
    "companion": CompanionOut,
}
VALID_ROLES = {"user", "assistant", "system", "developer"}
_LLM_ID_BY_CRIT = {v: k for k, v in LCD_CRITERION_BY_LLM_ID.items()}

_ADDRESS_RE = re.compile(r"\d+\s+[^,\n]+,\s*[^,\n]+,\s*[A-Za-z]{2}\.?\s+\d{5}(?:-\d{4})?")
_COST_RE = re.compile(r"afford|expensive|too much|can't pay|cannot pay", re.I)
_HUMAN_RE = re.compile(r"real person|a nurse|talk to someone|scared|fell|pain|hurt", re.I)
_YES_RE = re.compile(r"\b(yes|yeah|ok|okay|sure|works)\b", re.I)


def validate_request(agent: str, system: str, messages: Any, schema: type[BaseModel]) -> list[str]:
    errs: list[str] = []
    want = AGENT_SCHEMAS.get(agent)
    if want is None:
        errs.append(f"unknown agent {agent!r}")
    elif schema is not want:
        errs.append(f"{agent}: schema {schema.__name__} != {want.__name__}")
    if not isinstance(system, str) or not system.strip():
        errs.append(f"{agent}: empty instructions")
    if not isinstance(messages, list) or not messages:
        errs.append(f"{agent}: input must be a non-empty list")
    else:
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or set(m) != {"role", "content"}:
                errs.append(f"{agent}: input[{i}] must be {{role, content}}: {m!r}")
                continue
            if m["role"] not in VALID_ROLES:
                errs.append(f"{agent}: input[{i}] bad role {m['role']!r}")
            if not isinstance(m["content"], str) or not m["content"].strip():
                errs.append(f"{agent}: input[{i}] empty/non-str content")
    try:  # what client.responses.parse(text_format=schema) does before sending
        from openai.lib._parsing._responses import type_to_text_format_param

        fmt = type_to_text_format_param(schema)
        if fmt.get("type") != "json_schema" or not fmt.get("strict"):
            errs.append(f"{agent}: schema {schema.__name__} not strict json_schema")
    except ImportError:  # pragma: no cover
        pass
    except Exception as e:  # noqa: BLE001
        errs.append(f"{agent}: schema {schema.__name__} not convertible: {e}")
    return errs


class StubResponsesLLM:
    """LLMPort stand-in for OpenAILLM. `ledger` gets one LLMCall per request (token estimates, `model`)."""

    def __init__(
        self,
        scenario: Any = None,
        *,
        model: str = "gpt-4o-mini",
        fail_agents: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.scenario = scenario
        self.model = model
        self.fail_agents = set(fail_agents)
        self.ledger: list[LLMCall] = []
        self.requests: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.ctx: dict[tuple[str, str], Any] = {}  # (agent, key) -> view / (task, utterances, spec)
        self.spec_by_task: dict[str, Any] = {}

    # ------------------------------------------------------------------------------- port
    async def structured(self, *, agent: str, key: str, system: str, messages: list[dict], schema: type):
        self.requests.append({"agent": agent, "key": key, "system": system, "messages": list(messages)})
        self.errors += [f"{key}: {e}" for e in validate_request(agent, system, messages, schema)]
        if agent in self.fail_agents:
            self._log(agent, key, system, messages, None, ok=False)
            raise LLMUnavailable(f"{agent}:{key}: stub failure")
        try:
            out = await getattr(self, f"_{agent}")(key, system, messages)
        except Exception as e:  # stub bug or missing context -> record, surface as LLM failure
            self.errors.append(f"{agent}:{key}: stub raised {type(e).__name__}: {e}")
            raise
        out = schema.model_validate(out.model_dump())  # must round-trip through the schema
        self._log(agent, key, system, messages, out, ok=True)
        return out

    def _log(self, agent: str, key: str, system: str, messages: list[dict], out: Any, ok: bool) -> None:
        in_tok = (len(system) + len(json.dumps(messages))) // 4 + 1
        out_tok = (len(out.model_dump_json()) // 4 + 1) if out is not None else 0
        self.ledger.append(
            LLMCall(agent=agent, key=key, model=self.model, in_tok=in_tok, out_tok=out_tok, ms=1, ok=ok)
        )

    def calls_by_agent(self) -> Counter:
        return Counter(r["agent"] for r in self.requests)

    # ------------------------------------------------------------------------------- agents
    async def _caller(self, key: str, system: str, messages: list[dict]) -> CallerTurn:
        view = self.ctx[("caller", key)]
        return await FakeCallerBrain().next(view)

    async def _persona(self, key: str, system: str, messages: list[dict]) -> PersonaTurn:
        view, spec = self.ctx[("persona", key)]
        if spec.answerer_name not in system or spec.org_name not in system:
            self.errors.append(f"persona:{key}: system prompt does not carry the persona spec")
        return await FakePersonaBrain(spec).next(view)

    async def _extractor(self, key: str, system: str, messages: list[dict]) -> Extraction:
        task, utts = self.ctx[("extractor", key.rsplit(":", 1)[0])]
        spec = self.spec_by_task.get(task.task_id)
        content = messages[0]["content"]
        claims: dict[str, ClaimLLM] = {}
        for slot in task.slots:
            ask = normalize(slot.ask)
            answer = None
            for i, u in enumerate(utts):
                if u.speaker == "caller" and ask and ask in normalize(u.text):
                    nxt = next((v for v in utts[i + 1 :] if v.speaker == "counterparty"), None)
                    if nxt is not None and nxt.event not in ("transfer", "hold"):
                        answer = nxt
            if answer is None or answer.text not in content:
                continue
            fact = None
            if spec is not None:
                fact = next(
                    (
                        f
                        for f in spec.facts
                        if f.slot == slot.id and normalize(f.text) in normalize(answer.text)
                    ),
                    None,
                )
            if fact is None:
                continue
            claims[slot.id] = ClaimLLM(
                field=slot.id, status="stated", value=fact.value, evidence_quote=answer.text
            )
        fields = fields_for(task)
        if key.endswith("passB"):
            fields = list(reversed(fields))
        out = [
            claims.get(f) or ClaimLLM(field=f, status="unknown", value=None, evidence_quote=None)
            for f in fields
        ]
        return Extraction(claims=out)

    async def _lcd(self, key: str, system: str, messages: list[dict]) -> LCDCheck:
        m = re.search(r"<<<\n(.*)\n>>>", messages[0]["content"], re.S)
        note = m.group(1) if m else ""
        claims = []
        pcp = self.scenario.personas.get(PCP_PARTY_ID) if self.scenario is not None else None
        if pcp is not None:
            if pcp.sim.visit_note_text and normalize(pcp.sim.visit_note_text) == normalize(note):
                claims = list(pcp.sim.note_claims)
            elif pcp.sim.addendum_text and normalize(pcp.sim.addendum_text) == normalize(note):
                claims = list(pcp.sim.addendum_claims)
        by_crit = {}
        for c in claims:
            try:
                crit = LCDCriterion(c.field)
            except ValueError:
                crit = LCD_CRITERION_BY_LLM_ID.get(c.field)
            if crit is not None:
                by_crit[crit] = c
        results = []
        for crit in LCDCriterion:
            c = by_crit.get(crit)
            if c is None or c.status != "stated" or not c.evidence_quote:
                results.append(
                    CriterionResult(criterion_id=_LLM_ID_BY_CRIT[crit], met="not_documented", quote=None)
                )
            else:
                results.append(
                    CriterionResult(
                        criterion_id=_LLM_ID_BY_CRIT[crit], met=_lcd_value(c.value), quote=c.evidence_quote
                    )
                )
        return LCDCheck(results=results)

    async def _companion(self, key: str, system: str, messages: list[dict]) -> CompanionOut:
        if key.startswith("compose"):
            return CompanionOut(
                reply="Hello, here is a quick update about your wheelchair. "
                "A nurse is always available if you need one.",
                intents=[],
            )
        text = messages[-1]["content"]
        intents: list[IntentLLM] = []

        def add(kind: str, **kw: Any) -> None:
            intents.append(
                IntentLLM(
                    kind=kind,
                    address=kw.get("address"),
                    preferred_time=kw.get("preferred_time"),
                    note=kw.get("note"),
                )
            )

        m = _ADDRESS_RE.search(text)
        if m:
            add("confirm_address", address=m.group(0))
        if _COST_RE.search(text):
            add("cost_concern")
        if _HUMAN_RE.search(text):
            add("needs_human", note="human")
        if not intents and _YES_RE.search(text):
            add("give_consent")
        return CompanionOut(reply="Thank you, I've noted that.", intents=intents)


# ----------------------------------------------------------------------------------- install


def install(monkeypatch: Any, stub: StubResponsesLLM) -> StubResponsesLLM:
    """Make `make_llm("live")` return `stub` and give it the views the LLM brains see."""
    import miramace.llm.port as port

    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("MIRA_TEXT_MODEL", "dummy-model")
    monkeypatch.setattr(port, "make_llm", lambda mode=None: stub)

    orig_caller, orig_persona, orig_extract = (
        LLMCallerBrain.next,
        LLMPersonaBrain.next,
        Extractor.extract_verified,
    )

    async def caller_next(self, view):
        if self.llm is stub:
            stub.ctx[("caller", f"{view.task.task_id}:t{len(view.utterances)}")] = view
        return await orig_caller(self, view)

    async def persona_next(self, view):
        if self.llm is stub:
            stub.ctx[("persona", f"{view.task.task_id}:t{len(view.utterances)}")] = (view, self.spec)
            stub.spec_by_task[view.task.task_id] = self.spec
        return await orig_persona(self, view)

    async def extract(self, task, utterances):
        if self.llm is stub:
            stub.ctx[("extractor", task.task_id)] = (task, list(utterances))
        return await orig_extract(self, task, utterances)

    monkeypatch.setattr(LLMCallerBrain, "next", caller_next)
    monkeypatch.setattr(LLMPersonaBrain, "next", persona_next)
    monkeypatch.setattr(Extractor, "extract_verified", extract)
    return stub
