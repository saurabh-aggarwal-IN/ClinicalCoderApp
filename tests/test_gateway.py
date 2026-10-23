from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from clinical_coder.llm.budget import BudgetLimits, BudgetTracker
from clinical_coder.llm.client import LLMGateway
from clinical_coder.llm.pricing import PriceTable
from clinical_coder.llm.runtime import reset_runtimes

MESSAGES = [{"role": "user", "content": "Assign codes."}]


def gateway(cfg, handler, *, retries: int = 1, calls: int = 4) -> LLMGateway:
    cfg = replace(
        cfg,
        llm_max_retries=retries,
        llm_rpm=0,
        llm_max_concurrency=2,
    )
    budget = BudgetTracker(
        limits=BudgetLimits(
            max_llm_calls=calls,
            max_input_tokens=10_000,
            max_output_tokens=2_000,
            max_total_tokens=20_000,
            max_wall_seconds=20.0,
            max_cost_usd=None,
        ),
        model="mock/model",
        prices=PriceTable(),
    )
    gw = LLMGateway(
        cfg,
        budget,
        base_url=f"http://mock-{id(handler)}.local/v1",
        transport=httpx.MockTransport(handler),
    )
    gw._sleep_backoff = _no_sleep  # type: ignore[method-assign]
    return gw


async def _no_sleep(*_args, **_kwargs) -> None:
    return None


def completion(
    content: str = '{"codes":[]}', model: str = "mock/model"
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        },
    )


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_runtimes()


@pytest.mark.asyncio
async def test_successful_call_records_usage(cfg) -> None:
    gw = gateway(cfg, lambda request: completion())
    try:
        result = await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert result.content == '{"codes":[]}'
    assert result.usage.total_tokens == 30
    assert result.attempts == 1
    assert gw.budget.report().used["cost_known"] is False


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds(cfg) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="down")
        return completion()

    gw = gateway(cfg, handler)
    try:
        result = await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert result.attempts == 2
    assert result.retry_reasons == ["LLM_HTTP_503"]
    assert gw.budget.report().used["retries"] == 1


@pytest.mark.asyncio
async def test_non_retryable_4xx_is_not_retried(cfg) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    gw = gateway(cfg, handler)
    try:
        with pytest.raises(Exception) as excinfo:
            await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert getattr(excinfo.value, "code") == "LLM_HTTP_401"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_timeout_is_retried_then_structured(cfg) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    gw = gateway(cfg, handler)
    try:
        with pytest.raises(Exception) as excinfo:
            await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert getattr(excinfo.value, "code") == "LLM_TIMEOUT"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_empty_content_is_an_error(cfg) -> None:
    gw = gateway(cfg, lambda request: completion(content=""))
    try:
        with pytest.raises(Exception) as excinfo:
            await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert getattr(excinfo.value, "code") == "LLM_EMPTY_RESPONSE"


@pytest.mark.asyncio
async def test_budget_blocks_extra_calls(cfg) -> None:
    gw = gateway(cfg, lambda request: completion(), calls=1)
    try:
        await gw.complete(MESSAGES, model="mock/model")
        with pytest.raises(Exception) as excinfo:
            await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert getattr(excinfo.value, "code") == "BUDGET_EXCEEDED"


@pytest.mark.asyncio
async def test_circuit_breaker_opens(cfg) -> None:
    gw = gateway(cfg, lambda request: completion(), retries=0)
    try:
        for _ in range(5):
            await gw.runtime.breaker.record_failure()
        with pytest.raises(Exception) as excinfo:
            await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert getattr(excinfo.value, "code") == "CIRCUIT_OPEN"


@pytest.mark.asyncio
async def test_model_substitution_is_flagged(cfg) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return completion(model="mock/other-model")

    gw = gateway(cfg, handler)
    try:
        result = await gw.complete(MESSAGES, model="mock/model")
    finally:
        await gw.aclose()
    assert "MODEL_SUBSTITUTED" in result.guardrail_flags
