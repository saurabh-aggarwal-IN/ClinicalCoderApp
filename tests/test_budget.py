from __future__ import annotations

import pytest

from clinical_coder.contracts import BudgetSpec, Usage
from clinical_coder.llm.budget import (
    BudgetExceeded,
    BudgetLimits,
    BudgetTracker,
    estimate_tokens,
)
from clinical_coder.llm.pricing import Price, PriceTable


def tracker(cfg, **overrides) -> BudgetTracker:
    limits = BudgetLimits(
        max_llm_calls=overrides.get("max_llm_calls", 2),
        max_input_tokens=overrides.get("max_input_tokens", 1_000),
        max_output_tokens=overrides.get("max_output_tokens", 1_000),
        max_total_tokens=overrides.get("max_total_tokens", 2_000),
        max_wall_seconds=overrides.get("max_wall_seconds", 30.0),
        max_cost_usd=overrides.get("max_cost_usd", None),
    )
    return BudgetTracker(
        limits=limits,
        model="mock/model",
        prices=PriceTable(
            {"mock/model": Price(input_per_mtok=1.0, output_per_mtok=2.0)}
        ),
    )


def test_effective_budget_clamps_to_server(cfg) -> None:
    limits = BudgetLimits.effective(
        BudgetSpec(
            max_llm_calls=64,
            max_input_tokens=2_000_000,
            max_output_tokens=128_000,
            max_total_tokens=2_000_000,
            max_wall_seconds=3_600,
        ),
        cfg,
    )
    assert limits.max_llm_calls == cfg.server_max_llm_calls
    assert limits.max_input_tokens == cfg.server_max_input_tokens
    assert limits.max_output_tokens == cfg.server_max_output_tokens
    assert limits.max_total_tokens == cfg.server_max_total_tokens
    assert limits.max_wall_seconds == cfg.server_max_wall_seconds


def test_call_limit_is_enforced(cfg) -> None:
    b = tracker(cfg, max_llm_calls=1)
    b.before_call(10, 10)
    b.after_call(Usage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1))
    with pytest.raises(BudgetExceeded, match="max_llm_calls"):
        b.before_call(10, 10)


def test_token_limit_is_enforced_before_call(cfg) -> None:
    b = tracker(cfg, max_input_tokens=100, max_total_tokens=500)
    with pytest.raises(BudgetExceeded, match="max_input_tokens"):
        b.before_call(200, 10)


def test_output_limit_after_call(cfg) -> None:
    b = tracker(cfg, max_output_tokens=10, max_total_tokens=10_000)
    with pytest.raises(BudgetExceeded, match="max_output_tokens"):
        b.after_call(Usage(input_tokens=1, output_tokens=11, total_tokens=12, calls=1))


def test_cost_limit(cfg) -> None:
    b = tracker(cfg, max_cost_usd=0.000001)
    with pytest.raises(BudgetExceeded, match="max_cost_usd"):
        b.after_call(Usage(input_tokens=1_000, output_tokens=1_000, total_tokens=2_000, calls=1))


def test_failed_attempt_counts_estimated_input(cfg) -> None:
    b = tracker(cfg, max_llm_calls=2)
    b.before_call(100, 10)
    b.after_failed_attempt(100)
    report = b.report()
    assert report.used["input_tokens"] == 100
    assert report.used["retries"] == 1


def test_cost_is_none_when_prices_unknown(cfg) -> None:
    b = BudgetTracker(
        limits=tracker(cfg).limits,
        model="unknown/model",
        prices=PriceTable(),
    )
    b.after_call(Usage(input_tokens=10, output_tokens=10, total_tokens=20, calls=1))
    report = b.report()
    assert report.used["cost_known"] is False
    assert report.used["cost_usd"] is None


def test_estimate_tokens_is_monotonic() -> None:
    short = estimate_tokens([{"role": "user", "content": "a" * 40}])
    long = estimate_tokens([{"role": "user", "content": "a" * 400}])
    assert long > short
