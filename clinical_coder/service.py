from __future__ import annotations

import asyncio
import time
import uuid
from clinical_coder.contracts import (
    CodeRequest,
    CodeResponse,
    ErrorInfo,
    GuardrailReport,
    Usage,
)
from clinical_coder.guardrails import check_input, check_output
from clinical_coder.harnesses.base import BaseHarness, HarnessExecutionError
from clinical_coder.llm.budget import BudgetExceeded
from clinical_coder.llm.client import LLMError
from clinical_coder.output import OutputFormatError, is_em_code
from clinical_coder.settings import Settings
from clinical_coder.trace import TraceRecorder


class HarnessService:
    """Shared request lifecycle for every harness.

    Input guardrails, budget accounting, output validation, trace creation, and
    error normalization live here so all three implementations return the same
    contract even when a model, the CLI, or a graph fails.
    """

    def __init__(self, harness: BaseHarness, cfg: Settings) -> None:
        self.harness = harness
        self.cfg = cfg

    async def handle(self, request: CodeRequest) -> CodeResponse:
        started = time.monotonic()
        request_id = request.request_id or f"{self.harness.harness_id}-{uuid.uuid4().hex}"
        trace = TraceRecorder()
        budget = self.harness.new_budget(request)
        model = self.harness.model_for(request)

        codes = []
        raw_output: str | None = None
        usage = Usage()
        model_actual: str | None = None
        hermes_version: str | None = None
        warnings: list[str] = []
        error: ErrorInfo | None = None
        guardrails = GuardrailReport()

        input_report = check_input(request, self.cfg)
        guardrails.extend(input_report)
        trace.guardrail(input_report)

        if input_report.blocked:
            error = _guardrail_error(input_report)
            trace.error(error.code, error.message)
        else:
            try:
                result = await self.harness.run(request, budget, trace)
                codes = result.codes
                raw_output = result.raw_output
                usage = result.usage
                model_actual = result.model_actual
                hermes_version = result.hermes_version
                warnings.extend(result.warnings)
                guardrails.extend(result.guardrails)
            except HarnessExecutionError as exc:
                error, raw_output = _execution_error(exc, self.cfg)
                trace.error(error.code, error.message, **error.details)
            except BudgetExceeded as exc:
                error = ErrorInfo(
                    code="BUDGET_EXCEEDED",
                    message=f"Request budget exhausted: {exc.reason}",
                    retryable=False,
                    details=exc.report.model_dump(mode="json"),
                )
                trace.error(error.code, error.message)
            except LLMError as exc:
                error = ErrorInfo(
                    code=exc.code,
                    message=_scrub(exc, self.cfg),
                    retryable=exc.retryable,
                    details={k: v for k, v in exc.details.items() if k != "raw_output"},
                )
                trace.error(error.code, error.message, status=exc.status)
            except OutputFormatError as exc:
                error = ErrorInfo(
                    code="OUTPUT_FORMAT_INVALID",
                    message=str(exc),
                    retryable=False,
                    details=exc.details,
                )
                raw_output = str(exc.details.get("raw_output") or "") or None
                trace.error(error.code, error.message)
            except asyncio.TimeoutError as exc:
                error = ErrorInfo(
                    code="HARNESS_TIMEOUT",
                    message=f"The harness exceeded its wall-clock budget: {exc}",
                    retryable=True,
                )
                trace.error(error.code, error.message)
            except Exception as exc:  # noqa: BLE001 - contract must not leak tracebacks
                error = ErrorInfo(
                    code="HARNESS_ERROR",
                    message=_scrub(exc, self.cfg),
                    retryable=False,
                    details={"type": type(exc).__name__},
                )
                trace.error(error.code, error.message)

        if codes:
            output_report = check_output(codes, request, self.cfg)
            guardrails.extend(output_report)
            trace.guardrail(output_report)
            if output_report.blocked:
                error = _guardrail_error(output_report)
                trace.error(error.code, error.message)
                if self.cfg.strict_guardrails:
                    codes = []
            elif output_report.provenance_ok is False:
                warnings.append(
                    "At least one code has no verbatim supporting snippet; the "
                    "deterministic provenance grader will treat it as a miss."
                )
        elif error is None:
            error = ErrorInfo(
                code="NO_CODES",
                message="The harness returned no codes.",
                retryable=False,
            )
            trace.error(error.code, error.message)

        budget_report = budget.report()
        if budget_report.exceeded and error is None:
            error = ErrorInfo(
                code="BUDGET_EXCEEDED",
                message=f"Request budget exhausted: {budget_report.stop_reason}",
                retryable=False,
                details=budget_report.model_dump(mode="json"),
            )
            trace.error(error.code, error.message)
        trace.budget(budget_report)
        if usage.calls or usage.total_tokens:
            trace.usage(
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "total_tokens": usage.total_tokens,
                    "calls": usage.calls,
                    "cost_usd": budget_report.used.get("cost_usd"),
                    "cost_known": budget_report.used.get("cost_known", True),
                }
            )

        warnings = _dedupe(warnings)
        if codes and all(is_em_code(entry.code) for entry in codes):
            warnings.append("Every returned code is an E/M code; E/M assignment is out of scope.")

        return CodeResponse(
            request_id=request_id,
            case_id=request.case_id,
            harness=self.harness.info(
                model=model,
                model_actual=model_actual,
                hermes_version=hermes_version,
            ),
            codes=codes,
            trace=trace.to_list(),
            usage=usage,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            raw_output=_truncate(raw_output, self.cfg.max_raw_output_chars),
            error=error,
            guardrails=guardrails,
            budget=budget_report,
            warnings=warnings,
            disclaimer=self.cfg.disclaimer,
        )


def _execution_error(
    exc: HarnessExecutionError, cfg: Settings
) -> tuple[ErrorInfo, str | None]:
    details = dict(exc.details)
    raw_output = details.pop("raw_output", None)
    return (
        ErrorInfo(
            code=exc.code,
            message=_scrub(exc, cfg),
            retryable=exc.retryable,
            details=details,
        ),
        raw_output if isinstance(raw_output, str) else None,
    )


def _guardrail_error(report: GuardrailReport) -> ErrorInfo:
    codes = sorted({violation.code for violation in report.violations if violation.severity == "block"})
    return ErrorInfo(
        code="GUARDRAIL_BLOCKED",
        message="Input or output guardrails blocked this request.",
        retryable=False,
        details={"violations": codes},
    )


def _scrub(exc: BaseException, cfg: Settings) -> str:
    text = str(exc) or type(exc).__name__
    if cfg.llm_api_key:
        text = text.replace(cfg.llm_api_key, "[REDACTED]")
    return text[:2_000]


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
