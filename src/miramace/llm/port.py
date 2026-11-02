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


DEFAULT_MAX_LLM_CALLS = 400
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_SDK_RETRIES = 2


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


class OpenAILLM:
    """OpenAI Responses API with strict structured outputs. One repair retry on parse failure.

    Safety rails for real runs (env): MIRA_LLM_TIMEOUT (s per request, default 45), MIRA_LLM_RETRIES (SDK
    retries on 429/5xx/timeouts, default 2), MIRA_MAX_OUTPUT_TOKENS (default 1200; the repair retry gets 4x,
    since reasoning tokens count against the cap and truncation is the usual parse failure), and
    MIRA_MAX_LLM_CALLS (hard budget of API requests per OpenAILLM instance = per run, default 400; beyond it
    every call raises LLMUnavailable without touching the network).
    """

    def __init__(
        self, model: str | None = None, *, client: Any | None = None, max_calls: int | None = None
    ) -> None:
        self.model = model or os.environ.get("MIRA_TEXT_MODEL") or ""
        if not self.model:
            raise LLMUnavailable("MIRA_TEXT_MODEL is not set and no model was given")
        self.effort = os.environ.get("MIRA_REASONING_EFFORT") or None
        self.max_output_tokens = _env_int("MIRA_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        self.max_calls = (
            max_calls if max_calls is not None else _env_int("MIRA_MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS)
        )
        self.timeout = _env_float("MIRA_LLM_TIMEOUT", DEFAULT_TIMEOUT_S)
        self.sdk_retries = _env_int("MIRA_LLM_RETRIES", DEFAULT_SDK_RETRIES)
        self._client = client
        self.ledger: list[LLMCall] = []
        self.requests_made = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(timeout=self.timeout, max_retries=self.sdk_retries)
        return self._client

    async def structured(
        self, *, agent: str, key: str, system: str, messages: list[dict], schema: type[T]
    ) -> T:
        msgs = list(messages)
        last_err: Exception | None = None
        max_out = self.max_output_tokens
        for _attempt in range(2):  # first try + 1 repair retry
            if self.requests_made >= self.max_calls:
                raise LLMUnavailable(
                    f"{agent}:{key}: LLM call budget exhausted ({self.max_calls}; raise MIRA_MAX_LLM_CALLS "
                    "or --max-llm-calls)"
                )
            self.requests_made += 1
            t0 = time.perf_counter()
            in_tok = out_tok = 0
            try:
                resp = await self.client.responses.parse(
                    model=self.model,
                    instructions=system,
                    input=msgs,
                    text_format=schema,
                    store=False,
                    max_output_tokens=max_out,
                    **({"reasoning": {"effort": self.effort}} if self.effort else {}),
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
                    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
                parsed = getattr(resp, "output_parsed", None)
                if parsed is None:
                    details = getattr(resp, "incomplete_details", None)
                    why = getattr(details, "reason", None) if details is not None else None
                    raise ValueError(f"model returned no parsed output ({why or 'refusal or truncation'})")
                if not isinstance(parsed, schema):
                    parsed = schema.model_validate(parsed)
                self._log(agent, key, t0, in_tok, out_tok, True)
                return parsed
            except (ValidationError, ValueError) as e:  # parse problem -> repair once
                self._log(agent, key, t0, in_tok, out_tok, False)
                last_err = e
                max_out = max_out * 4  # truncation (incl. reasoning tokens) is the usual cause
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
        missing = [k for k in ("OPENAI_API_KEY", "MIRA_TEXT_MODEL") if not os.environ.get(k)]
        if missing:
            raise LLMUnavailable(
                f"live LLM mode requires {' and '.join(missing)} (set in .env or the environment)"
            )
        return OpenAILLM()
    raise ValueError(f"unknown LLM mode {mode!r} (expected 'fake' or 'live')")
