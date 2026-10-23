from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from clinical_coder.contracts import BudgetReport, BudgetSpec, Usage
from clinical_coder.llm.pricing import PriceTable
from clinical_coder.settings import Settings


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, report: BudgetReport) -> None:
        super().__init__(reason)
        self.code = "BUDGET_EXCEEDED"
        self.reason = reason
        self.report = report


@dataclass
class BudgetLimits:
    max_llm_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_wall_seconds: float
    max_cost_usd: float | None

    @classmethod
    def effective(cls, request: BudgetSpec, cfg: Settings) -> "BudgetLimits":
        server_cost = cfg.server_max_cost_usd
        requested_cost = request.max_cost_usd
        if requested_cost is None:
            cost = server_cost
        elif server_cost is None:
            cost = requested_cost
        else:
            cost = min(requested_cost, server_cost)
        return cls(
            max_llm_calls=min(request.max_llm_calls, cfg.server_max_llm_calls),
            max_input_tokens=min(request.max_input_tokens, cfg.server_max_input_tokens),
            max_output_tokens=min(request.max_output_tokens, cfg.server_max_output_tokens),
            max_total_tokens=min(request.max_total_tokens, cfg.server_max_total_tokens),
            max_wall_seconds=min(request.max_wall_seconds, cfg.server_max_wall_seconds),
            max_cost_usd=cost,
        )


@dataclass
class BudgetTracker:
    limits: BudgetLimits
    model: str
    prices: PriceTable
    started_at: float = field(default_factory=time.monotonic)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    exceeded: bool = False
    stop_reason: str | None = None
    retries: int = 0

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_wall_s(self) -> float:
        return max(0.0, self.limits.max_wall_seconds - self.elapsed_s())

    def check_wall(self) -> None:
        if self.elapsed_s() > self.limits.max_wall_seconds:
            self._exceed("max_wall_seconds")

    def _exceed(self, reason: str) -> None:
        self.exceeded = True
        self.stop_reason = self.stop_reason or reason
        raise BudgetExceeded(reason, self.report())

    def before_call(self, estimated_input_tokens: int, max_output_tokens: int) -> None:
        self.check_wall()
        if self.calls >= self.limits.max_llm_calls:
            self._exceed("max_llm_calls")
        if max_output_tokens > self.limits.max_output_tokens:
            self._exceed("max_output_tokens")
        if estimated_input_tokens > self.limits.max_input_tokens:
            self._exceed("max_input_tokens")
        if self.input_tokens + estimated_input_tokens > self.limits.max_input_tokens:
            self._exceed("max_input_tokens")
        if self.total_tokens + estimated_input_tokens + max_output_tokens > self.limits.max_total_tokens:
            self._exceed("max_total_tokens")
        self.calls += 1

    def after_failed_attempt(self, estimated_input_tokens: int) -> None:
        """Account for a provider request that failed before returning usage.

        Providers can still bill the prompt when a stream dies, so the estimate
        is counted conservatively. The call itself was already counted by
        ``before_call``.
        """
        self.input_tokens += max(0, estimated_input_tokens)
        self.total_tokens += max(0, estimated_input_tokens)
        self.retries += 1
        self.check_wall()
        if self.input_tokens > self.limits.max_input_tokens:
            self._exceed("max_input_tokens")
        if self.total_tokens > self.limits.max_total_tokens:
            self._exceed("max_total_tokens")

    def after_call(self, usage: Usage, *, retry: bool = False) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        cost = self.prices.cost_usd(self.model, usage.input_tokens, usage.output_tokens)
        if cost is None:
            self.cost_known = False
        else:
            self.cost_usd += cost
            if self.limits.max_cost_usd is not None and self.cost_usd > self.limits.max_cost_usd:
                self._exceed("max_cost_usd")
        if retry:
            self.retries += 1
        self.check_wall()
        if self.input_tokens > self.limits.max_input_tokens:
            self._exceed("max_input_tokens")
        if self.output_tokens > self.limits.max_output_tokens:
            self._exceed("max_output_tokens")
        if self.total_tokens > self.limits.max_total_tokens:
            self._exceed("max_total_tokens")

    def report(self) -> BudgetReport:
        return BudgetReport(
            limits={
                "max_llm_calls": self.limits.max_llm_calls,
                "max_input_tokens": self.limits.max_input_tokens,
                "max_output_tokens": self.limits.max_output_tokens,
                "max_total_tokens": self.limits.max_total_tokens,
                "max_wall_seconds": self.limits.max_wall_seconds,
                "max_cost_usd": self.limits.max_cost_usd,
            },
            used={
                "llm_calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost_usd if self.cost_known else None,
                "cost_known": self.cost_known,
                "retries": self.retries,
                "elapsed_s": round(self.elapsed_s(), 3),
            },
            exceeded=self.exceeded,
            stop_reason=self.stop_reason,
        )


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        chars += len(str(message.get("role", "")))
        chars += len(str(message.get("content", "")))
        chars += 8
    return max(1, chars // 4)
