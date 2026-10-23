from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from clinical_coder.contracts import (
    CodeEntry,
    CodeRequest,
    GuardrailReport,
    HarnessInfo,
    Usage,
)
from clinical_coder.llm.budget import BudgetLimits, BudgetTracker, estimate_tokens
from clinical_coder.llm.pricing import PriceTable
from clinical_coder.settings import Settings
from clinical_coder.trace import TraceRecorder


class HarnessExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class GuardrailBlocked(HarnessExecutionError):
    def __init__(self, message: str, report: GuardrailReport) -> None:
        super().__init__("GUARDRAIL_BLOCKED", message)
        self.report = report


@dataclass
class HarnessResult:
    """What a harness returns before the shared service applies final guards."""

    codes: list[CodeEntry]
    raw_output: str
    usage: Usage
    model_actual: str | None = None
    hermes_version: str | None = None
    guardrails: GuardrailReport = field(default_factory=GuardrailReport)
    warnings: list[str] = field(default_factory=list)


class BaseHarness(ABC):
    harness_id: ClassVar[str]
    harness_name: ClassVar[str]

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

    def model_for(self, request: CodeRequest) -> str:
        return (request.model or self.cfg.llm_model).strip()

    def info(
        self,
        *,
        model: str,
        model_actual: str | None = None,
        hermes_version: str | None = None,
    ) -> HarnessInfo:
        return HarnessInfo(
            id=self.harness_id,
            name=self.harness_name,
            version=self.cfg.harness_version,
            commit=self.cfg.git_commit,
            model=model,
            model_actual=model_actual,
            implementation=self.cfg.implementation,
            hermes_version=hermes_version,
        )

    def new_budget(self, request: CodeRequest) -> BudgetTracker:
        limits = BudgetLimits.effective(request.budget, self.cfg)
        return BudgetTracker(
            limits=limits,
            model=self.model_for(request),
            prices=PriceTable.from_env(self.cfg.llm_prices_file or None),
        )

    def estimate_input_tokens(self, *texts: str) -> int:
        return estimate_tokens(
            [{"role": "system", "content": text} for text in texts if text]
        )

    @abstractmethod
    async def run(
        self,
        request: CodeRequest,
        budget: BudgetTracker,
        trace: TraceRecorder,
    ) -> HarnessResult:
        """Execute the harness and return parsed codes plus usage."""
