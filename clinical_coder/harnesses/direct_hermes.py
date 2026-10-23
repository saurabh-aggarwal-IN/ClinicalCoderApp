from __future__ import annotations

from typing import Any

from clinical_coder.contracts import CodeRequest
from clinical_coder.harnesses.base import (
    BaseHarness,
    HarnessExecutionError,
    HarnessResult,
)
from clinical_coder.hermes.runner import HermesRunner, HermesTimeout, HermesUnavailable
from clinical_coder.llm.budget import BudgetExceeded, BudgetTracker
from clinical_coder.output import OutputFormatError, parse_coding_output
from clinical_coder.settings import Settings
from clinical_coder.trace import TraceRecorder


def build_h1_query(request: CodeRequest) -> str:
    """H1 receives exactly the note and the task prompt, never history."""
    return (
        f"{request.prompt.strip()}\n\n"
        "The complete contents of the synthetic encounter note are included "
        "below. This run has no tools or file access, so use only this text.\n\n"
        "<encounter_note>\n"
        f"{request.encounter_note.strip()}\n"
        "</encounter_note>\n"
    )


class DirectHermesHarness(BaseHarness):
    harness_id = "h1"
    harness_name = "Direct Hermes"

    def __init__(self, cfg: Settings, runner: HermesRunner | None = None) -> None:
        super().__init__(cfg)
        self.runner = runner or HermesRunner(cfg)

    async def run(
        self,
        request: CodeRequest,
        budget: BudgetTracker,
        trace: TraceRecorder,
    ) -> HarnessResult:
        model = self.model_for(request)
        query = build_h1_query(request)
        max_turns = min(
            request.max_turns or self.cfg.hermes_max_turns,
            budget.limits.max_llm_calls,
        )
        timeout_s = min(
            self.cfg.request_timeout_s,
            max(1.0, budget.remaining_wall_s() - 1.0),
        )

        trace.turn_start(
            harness=self.harness_id,
            model=model,
            max_turns=max_turns,
            timeout_s=round(timeout_s, 3),
            query_chars=len(query),
            history_included=False,
        )

        estimated_input = self.estimate_input_tokens(query, "icd-cpt-coding skill rules")
        try:
            budget.before_call(
                estimated_input,
                min(4_000, budget.limits.max_output_tokens),
            )
        except BudgetExceeded as exc:
            raise HarnessExecutionError(
                "BUDGET_EXCEEDED",
                f"Budget rejected the Hermes call: {exc.reason}",
                details=exc.report.model_dump(mode="json"),
            ) from exc

        try:
            result = await self.runner.run(
                query=query,
                encounter_note=request.encounter_note,
                model=model,
                max_turns=max_turns,
                timeout_s=timeout_s,
            )
        except HermesTimeout as exc:
            raise HarnessExecutionError(
                "HERMES_TIMEOUT", str(exc), retryable=True
            ) from exc
        except HermesUnavailable as exc:
            raise HarnessExecutionError("HERMES_UNAVAILABLE", str(exc)) from exc

        trace.usage(
            {
                "provider": "hermes",
                "model": model,
                "session_id": result.session_id,
                "duration_ms": result.duration_ms,
                "exit_code": result.exit_code,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        )
        self._trace_hermes_events(result.events, trace)

        if result.error:
            raise HarnessExecutionError(
                "HERMES_FAILED",
                _redact(result.error, self.cfg.llm_api_key),
                details={
                    "exit_code": result.exit_code,
                    "stderr_tail": _redact(result.stderr[-2_000:], self.cfg.llm_api_key),
                },
            )

        if not result.raw_output:
            raise HarnessExecutionError(
                "HERMES_EMPTY_RESPONSE",
                "Hermes completed without a final response.",
                details={"exit_code": result.exit_code},
            )

        try:
            codes = parse_coding_output(result.raw_output)
        except OutputFormatError as exc:
            raise HarnessExecutionError(
                "OUTPUT_FORMAT_INVALID",
                str(exc),
                details={**exc.details, "raw_output": result.raw_output},
            ) from exc

        try:
            budget.after_call(result.usage)
        except BudgetExceeded as exc:
            raise HarnessExecutionError(
                "BUDGET_EXCEEDED",
                f"Hermes output exceeded the request budget: {exc.reason}",
                details={**exc.report.model_dump(mode="json"), "raw_output": result.raw_output},
            ) from exc

        trace.message(result.raw_output[:20_000], role="assistant", harness=self.harness_id)
        trace.turn_end(harness=self.harness_id, code_count=len(codes))

        return HarnessResult(
            codes=codes,
            raw_output=result.raw_output,
            usage=result.usage,
            model_actual=model,
            hermes_version=await self.runner.version(),
            warnings=[
                warning
                for warning in [
                    "Hermes ran with platform toolsets disabled."
                    if not self.cfg.hermes_toolsets
                    else "",
                    "H1 does not receive the patient history fixture."
                    if request.history is not None
                    else "",
                ]
                if warning
            ],
        )

    @staticmethod
    def _trace_hermes_events(events: list[dict[str, Any]], trace: TraceRecorder) -> None:
        for event in events:
            kind = str(event.get("type") or "")
            if kind in {"tool_call", "tool_use"}:
                trace.tool_call(
                    str(event.get("name") or event.get("tool") or "unknown"),
                    event.get("input") if isinstance(event.get("input"), dict) else {},
                )
            elif kind in {"tool_result", "tool_output"}:
                trace.tool_result(
                    str(event.get("name") or event.get("tool") or "unknown"),
                    event.get("output"),
                )


def _redact(text: str, secret: str) -> str:
    if secret:
        return text.replace(secret, "[REDACTED]")
    return text
