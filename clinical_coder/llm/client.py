from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from clinical_coder.contracts import Usage
from clinical_coder.llm.budget import BudgetExceeded, BudgetTracker, estimate_tokens
from clinical_coder.llm.pricing import PriceTable
from clinical_coder.llm.runtime import (
    CircuitOpenError,
    ProviderRuntime,
    get_runtime,
)
from clinical_coder.settings import Settings

logger = logging.getLogger("clinical_coder.llm")


class LLMError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.details = details or {}


@dataclass
class LLMResponse:
    content: str
    model: str
    model_actual: str
    usage: Usage
    attempts: int
    retry_reasons: list[str]
    latency_ms: int
    request_id: str | None
    guardrail_flags: list[str] = field(default_factory=list)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _usage_from_payload(payload: dict[str, Any]) -> Usage:
    raw = payload.get("usage") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    input_tokens = int(raw.get("prompt_tokens") or 0)
    output_tokens = int(raw.get("completion_tokens") or 0)
    cache_read = int(prompt_details.get("cached_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    total = int(raw.get("total_tokens") or 0)
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        reasoning_tokens=reasoning,
        total_tokens=total,
        calls=1,
    )
    usage.total_tokens = usage.total_tokens or (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.reasoning_tokens
    )
    return usage


class LLMGateway:
    """One guarded path for every direct model call.

    Retries, budget, rate limiting, and the circuit breaker are enforced here so
    a harness cannot accidentally bypass them.
    """

    def __init__(
        self,
        cfg: Settings,
        budget: BudgetTracker,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.cfg = cfg
        self.budget = budget
        self.base_url = (base_url or cfg.llm_base_url).rstrip("/")
        self.endpoint = (
            f"{self.base_url}/chat/completions"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/chat/completions"
        )
        self.api_key = api_key or cfg.llm_api_key
        self.transport = transport
        self.runtime: ProviderRuntime = get_runtime(
            self.base_url, cfg.llm_max_concurrency, cfg.llm_rpm
        )
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self.cfg.llm_timeout_s,
                connect=min(10.0, self.cfg.llm_timeout_s),
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
                limits=httpx.Limits(
                    max_connections=self.cfg.llm_max_concurrency * 2,
                    max_keepalive_connections=self.cfg.llm_max_concurrency,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4_000,
        request_id: str | None = None,
    ) -> LLMResponse:
        estimated_input = estimate_tokens(messages)
        attempts = 0
        retry_reasons: list[str] = []
        started = time.monotonic()
        max_attempts = self.cfg.llm_max_retries + 1
        # A caller may ask for more completion room than the request budget
        # allows; clamp rather than rejecting an otherwise valid call.
        max_tokens = max(1, min(max_tokens, self.budget.limits.max_output_tokens))

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.cfg.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if request_id:
            headers["X-Request-Id"] = request_id

        last_error: LLMError | None = None
        while attempts < max_attempts:
            attempts += 1
            try:
                self.budget.before_call(estimated_input, max_tokens)
            except BudgetExceeded as exc:
                raise LLMError(
                    "BUDGET_EXCEEDED",
                    f"Token/call budget exceeded before the model call: {exc.reason}",
                    details=exc.report.model_dump(mode="json"),
                ) from exc

            try:
                await self.runtime.breaker.before_call()
            except CircuitOpenError as exc:
                raise LLMError(
                    "CIRCUIT_OPEN",
                    "The model provider circuit breaker is open.",
                    retryable=True,
                    details={"retry_after_s": exc.retry_after_s},
                ) from exc

            try:
                async with self.runtime.limiter:
                    response = await self._http().post(
                        self.endpoint, json=payload, headers=headers
                    )
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_error = LLMError(
                    "LLM_TIMEOUT" if isinstance(exc, httpx.TimeoutException) else "LLM_NETWORK",
                    f"Model request failed: {type(exc).__name__}",
                    retryable=True,
                    details={"attempt": attempts},
                )
                retry_reasons.append(last_error.code)
                await self.runtime.breaker.record_failure()
                self.budget.after_failed_attempt(estimated_input)
                if attempts < max_attempts:
                    await self._sleep_backoff(attempts, None)
                    continue
                raise last_error from exc

            if response.status_code >= 200 and response.status_code < 300:
                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = LLMError(
                        "LLM_INVALID_RESPONSE",
                        "Model endpoint returned non-JSON.",
                        retryable=True,
                        status=response.status_code,
                    )
                    retry_reasons.append(last_error.code)
                    await self.runtime.breaker.record_failure()
                    self.budget.after_failed_attempt(estimated_input)
                    if attempts < max_attempts:
                        await self._sleep_backoff(attempts, None)
                        continue
                    raise last_error from exc

                choices = data.get("choices") or []
                if not choices:
                    last_error = LLMError(
                        "LLM_INVALID_RESPONSE",
                        "Model response has no choices.",
                        retryable=True,
                        status=response.status_code,
                    )
                    retry_reasons.append(last_error.code)
                    await self.runtime.breaker.record_failure()
                    self.budget.after_failed_attempt(estimated_input)
                    if attempts < max_attempts:
                        await self._sleep_backoff(attempts, None)
                        continue
                    raise last_error

                message = choices[0].get("message") or {}
                content = _content_text(message.get("content")).strip()
                if not content:
                    last_error = LLMError(
                        "LLM_EMPTY_RESPONSE",
                        "Model returned an empty message.",
                        retryable=True,
                        status=response.status_code,
                    )
                    retry_reasons.append(last_error.code)
                    await self.runtime.breaker.record_failure()
                    self.budget.after_failed_attempt(estimated_input)
                    if attempts < max_attempts:
                        await self._sleep_backoff(attempts, None)
                        continue
                    raise last_error

                usage = _usage_from_payload(data)
                actual_model = str(data.get("model") or model)
                guardrail_flags: list[str] = []
                if actual_model != model and not actual_model.endswith(model):
                    guardrail_flags.append("MODEL_SUBSTITUTED")
                await self.runtime.breaker.record_success()
                self.budget.after_call(usage)
                return LLMResponse(
                    content=content,
                    model=model,
                    model_actual=actual_model,
                    usage=usage,
                    attempts=attempts,
                    retry_reasons=retry_reasons,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    request_id=request_id,
                    guardrail_flags=guardrail_flags,
                )

            retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
            last_error = LLMError(
                f"LLM_HTTP_{response.status_code}",
                f"Model endpoint returned HTTP {response.status_code}.",
                retryable=retryable,
                status=response.status_code,
                details={"attempt": attempts, "body_preview": response.text[:300]},
            )
            if retryable:
                retry_reasons.append(last_error.code)
                # For temporary 429 rate limit cooldowns, avoid recording circuit breaker failures
                # on intermediate attempts so backoff has time to wait out the provider cooldown.
                # Only record breaker failure on hard errors (5xx/timeouts) or when all retries are exhausted.
                if response.status_code != 429 or attempts >= max_attempts:
                    await self.runtime.breaker.record_failure()
                self.budget.after_failed_attempt(estimated_input)
                if attempts < max_attempts:
                    delay_s = self._extract_retry_delay(response)
                    await self._sleep_backoff(attempts, delay_s)
                    continue
            raise last_error

        raise last_error or LLMError("LLM_UNKNOWN", "Model request failed")

    async def _sleep_backoff(self, attempt: int, delay_s: float | None) -> None:
        if delay_s is None:
            base = min(8.0, 0.5 * (2 ** (attempt - 1)))
            delay = base + random.uniform(0.0, base * 0.25)
        else:
            # When the model provider or router supplies an explicit reset/cooldown timestamp,
            # sleep for that duration plus a small jitter buffer (1.0-2.5s) to avoid race conditions.
            delay = delay_s + random.uniform(1.0, 2.5)
        delay = min(60.0, max(0.5, delay))
        remaining = self.budget.limits.max_wall_seconds - self.budget.elapsed_s()
        if remaining <= 0:
            raise LLMError(
                "BUDGET_EXCEEDED",
                "Wall-clock budget exhausted while backing off.",
                details=self.budget.report().model_dump(mode="json"),
            )
        await asyncio.sleep(min(delay, remaining))
        self.budget.check_wall()

    @classmethod
    def _extract_retry_delay(cls, response: httpx.Response) -> float | None:
        # 1. Standard Retry-After header
        header_val = response.headers.get("Retry-After")
        delay = cls._retry_after_seconds(header_val)
        if delay is not None:
            return delay

        # 2. Body inspection
        try:
            data = response.json()
        except Exception:
            data = None

        if isinstance(data, dict):
            err = data.get("error") if isinstance(data.get("error"), dict) else data

            # Epoch milliseconds retryAtMs (e.g. {"retryAtMs": 1792911522446})
            retry_at_ms = err.get("retryAtMs") or data.get("retryAtMs")
            if isinstance(retry_at_ms, (int, float)) and retry_at_ms > 0:
                target_epoch = retry_at_ms / 1000.0 if retry_at_ms > 1e11 else float(retry_at_ms)
                diff = target_epoch - time.time()
                return max(1.0, diff)

            # Explicit retry_after or retry_after_s
            for key in ("retry_after", "retry_after_s", "retryAfter"):
                val = err.get(key) or data.get(key)
                if val is not None:
                    try:
                        return max(1.0, float(val))
                    except (ValueError, TypeError):
                        pass

            # retry_after_ms
            retry_ms = err.get("retry_after_ms") or data.get("retry_after_ms")
            if isinstance(retry_ms, (int, float)) and retry_ms > 0:
                return max(1.0, retry_ms / 1000.0)

            # Text message regex search (e.g. "Soonest reset ~29s" or "retry after 15 seconds")
            msg = str(err.get("message") or "")
            match = re.search(r"(?:reset|cooldown|retry.*?after)\s*~?\s*(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
            if match:
                return max(1.0, float(match.group(1)))

        # 3. Fallback regex on raw text preview
        if response.text:
            match = re.search(r"(?:reset|cooldown|retry.*?after)\s*~?\s*(\d+(?:\.\d+)?)\s*s", response.text[:500], re.IGNORECASE)
            if match:
                return max(1.0, float(match.group(1)))

        return None

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None


def log_call(
    *,
    request_id: str | None,
    harness: str,
    model: str,
    outcome: str,
    latency_ms: int,
    attempts: int,
    usage: Usage | None = None,
    error: str | None = None,
) -> None:
    logger.info(
        json.dumps(
            {
                "event": "llm_call",
                "request_id": request_id,
                "harness": harness,
                "model": model,
                "outcome": outcome,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "error": error,
            },
            separators=(",", ":"),
        )
    )
