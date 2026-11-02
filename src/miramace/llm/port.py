"""B3 AI calls — OpenAILLM, ScriptedLLM, usage ledger, PRICES.

Both LLMs implement `ports.LLMPort.structured` and append one `LLMCall` per attempt to `self.ledger`.
`make_llm(mode)` picks the implementation from `MIRA_LLM_MODE` (fake|live): fake → None (pure-fake runs use
rule-based brains / fact-sheet fakes), live → OpenAILLM (requires OPENAI_API_KEY).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from miramace.models import LLMCall

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Only well-known public list prices; unknown model -> cost None ("$?").
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
_FREE_MODELS = {"scripted", "fake"}

DEFAULT_MAX_OUTPUT_TOKENS = 1200


class LLMUnavailable(RuntimeError):
    """Raised when the live LLM cannot produce a valid structured output (after 1 repair retry)."""


def cost_usd(ledger: list[LLMCall]) -> float | None:
    """Total cost of a ledger; None if any call used a model without a known price."""
    total = 0.0
    for c in ledger:
        if c.model in _FREE_MODELS:
            continue
        price = PRICES.get(c.model)
        if price is None:
            return None
        total += c.in_tok / 1e6 * price[0] + c.out_tok / 1e6 * price[1]
    return round(total, 6)


# ------------------------------------------------------------------------------------------- live


class OpenAILLM:
    """OpenAI Responses API with strict structured outputs. One repair retry on parse failure."""

    def __init__(self, model: str | None = None, *, client: Any | None = None) -> None:
        self.model = model or os.environ.get("MIRA_TEXT_MODEL") or ""
        if not self.model:
            raise LLMUnavailable("MIRA_TEXT_MODEL is not set and no model was given")
        self.effort = os.environ.get("MIRA_REASONING_EFFORT") or None
        self._client = client
        self.ledger: list[LLMCall] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(timeout=30, max_retries=2)
        return self._client

    async def structured(
        self, *, agent: str, key: str, system: str, messages: list[dict], schema: type[T]
    ) -> T:
        msgs = list(messages)
        last_err: Exception | None = None
        for _attempt in range(2):  # first try + 1 repair retry
            t0 = time.perf_counter()
            in_tok = out_tok = 0
            try:
                resp = await self.client.responses.parse(
                    model=self.model,
                    instructions=system,
                    input=msgs,
                    text_format=schema,
                    store=False,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    **({"reasoning": {"effort": self.effort}} if self.effort else {}),
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                parsed = getattr(resp, "output_parsed", None)
                if parsed is None:
                    raise ValueError("model returned no parsed output (refusal or truncation)")
                if not isinstance(parsed, schema):
                    parsed = schema.model_validate(parsed)
                self._log(agent, key, t0, in_tok, out_tok, True)
                return parsed
            except (ValidationError, ValueError) as e:  # parse problem -> repair once
                self._log(agent, key, t0, in_tok, out_tok, False)
                last_err = e
                msgs = list(messages) + [
                    {
                        "role": "user",
                        "content": f"Your previous output was invalid ({str(e)[:200]}). "
                        "Reply again with ONLY valid JSON matching the schema.",
                    }
                ]
            except Exception as e:  # network / API error (client already retried)
                self._log(agent, key, t0, in_tok, out_tok, False)
                raise LLMUnavailable(f"{agent}:{key}: {type(e).__name__}: {e}") from e
        raise LLMUnavailable(f"{agent}:{key}: invalid structured output after repair: {last_err}")

    def _log(self, agent: str, key: str, t0: float, in_tok: int, out_tok: int, ok: bool) -> None:
        ms = int((time.perf_counter() - t0) * 1000)
        self.ledger.append(
            LLMCall(agent=agent, key=key, model=self.model, in_tok=in_tok, out_tok=out_tok, ms=ms, ok=ok)
        )


# ----------------------------------------------------------------------------------------- scripted

Script = BaseModel | Callable[[list[dict]], BaseModel]


class ScriptedLLM:
    """Offline LLM. scripts[(agent, key)] -> model instance or callable(messages) -> model.

    Lookup falls back to (agent, "*") when present; otherwise a missing key raises KeyError.
    Every request is also kept in `requests` (agent, key, system, messages) for assertions.
    """

    def __init__(self, scripts: dict[tuple[str, str], Script] | None = None) -> None:
        self.scripts: dict[tuple[str, str], Script] = dict(scripts or {})
        self.ledger: list[LLMCall] = []
        self.requests: list[dict[str, Any]] = []

    async def structured(
        self, *, agent: str, key: str, system: str, messages: list[dict], schema: type[T]
    ) -> T:
        self.requests.append({"agent": agent, "key": key, "system": system, "messages": list(messages)})
        if (agent, key) in self.scripts:
            item = self.scripts[(agent, key)]
        elif (agent, "*") in self.scripts:
            item = self.scripts[(agent, "*")]
        else:
            raise KeyError((agent, key))
        out = item(messages) if callable(item) and not isinstance(item, BaseModel) else item
        if not isinstance(out, schema):
            out = schema.model_validate(out.model_dump() if isinstance(out, BaseModel) else out)
        self.ledger.append(
            LLMCall(agent=agent, key=key, model="scripted", in_tok=0, out_tok=0, ms=0, ok=True)
        )
        return out


# ------------------------------------------------------------------------------------------ factory


def make_llm(mode: str | None = None) -> OpenAILLM | None:
    """fake -> None (callers use fakes); live -> OpenAILLM (raises LLMUnavailable without a key)."""
    mode = (mode or os.environ.get("MIRA_LLM_MODE") or "fake").strip().lower()
    if mode == "fake":
        return None
    if mode == "live":
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMUnavailable("MIRA_LLM_MODE=live requires OPENAI_API_KEY")
        return OpenAILLM()
    raise ValueError(f"unknown LLM mode {mode!r} (expected 'fake' or 'live')")
