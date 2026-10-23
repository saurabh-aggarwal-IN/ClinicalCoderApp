from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_NAME = "clinical-coding-harness-v1"
CONTRACT_VERSION = "1.0.0"
DISCLAIMER = (
    "Synthetic clinical coding evaluation only. Not clinical advice, not a "
    "billing determination, and not validated for patient care."
)


class BudgetSpec(BaseModel):
    """Per-request ceiling. The server clamps every value to its own maximum."""

    model_config = ConfigDict(extra="forbid")

    max_llm_calls: int = Field(default=6, ge=1, le=64)
    max_input_tokens: int = Field(default=60_000, ge=100, le=2_000_000)
    max_output_tokens: int = Field(default=8_000, ge=64, le=128_000)
    max_total_tokens: int = Field(default=80_000, ge=100, le=2_000_000)
    max_wall_seconds: float = Field(default=120.0, ge=1.0, le=3_600.0)
    max_cost_usd: float | None = Field(default=None, ge=0.0, le=10_000.0)


class CodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, max_length=200)
    case_id: str = Field(min_length=1, max_length=100)
    encounter_note: str = Field(min_length=1)
    history: dict[str, Any] | None = None
    prompt: str = Field(min_length=1)
    model: str | None = Field(default=None, max_length=200)
    max_turns: int | None = Field(default=None, ge=1, le=100)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=2, max_length=20)
    system: Literal["ICD-10-CM", "CPT"]
    snippet: str = Field(min_length=3, max_length=500)
    rationale: str = Field(min_length=1, max_length=2_000)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        calls: int = 1,
    ) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.cache_read_tokens += max(0, cache_read_tokens)
        self.cache_write_tokens += max(0, cache_write_tokens)
        self.reasoning_tokens += max(0, reasoning_tokens)
        self.calls += max(0, calls)
        self.total_tokens = (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.reasoning_tokens
        )


TraceType = Literal[
    "turn_start",
    "message",
    "tool_call",
    "tool_result",
    "history_read",
    "guardrail",
    "usage",
    "budget",
    "error",
    "turn_end",
]


class TraceEvent(BaseModel):
    seq: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    type: TraceType
    payload: dict[str, Any] = Field(default_factory=dict)


class GuardrailViolation(BaseModel):
    code: str
    severity: Literal["block", "warn"]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class GuardrailReport(BaseModel):
    blocked: bool = False
    input_flags: list[str] = Field(default_factory=list)
    output_flags: list[str] = Field(default_factory=list)
    violations: list[GuardrailViolation] = Field(default_factory=list)
    phi_detected: bool = False
    prompt_injection_detected: bool = False
    key_leakage_detected: bool = False
    provenance_ok: bool | None = None

    def extend(self, other: "GuardrailReport") -> None:
        self.blocked = self.blocked or other.blocked
        self.input_flags.extend(other.input_flags)
        self.output_flags.extend(other.output_flags)
        self.violations.extend(other.violations)
        self.phi_detected = self.phi_detected or other.phi_detected
        self.prompt_injection_detected = (
            self.prompt_injection_detected or other.prompt_injection_detected
        )
        self.key_leakage_detected = (
            self.key_leakage_detected or other.key_leakage_detected
        )
        if other.provenance_ok is not None:
            self.provenance_ok = other.provenance_ok


class BudgetReport(BaseModel):
    limits: dict[str, Any] = Field(default_factory=dict)
    used: dict[str, Any] = Field(default_factory=dict)
    exceeded: bool = False
    stop_reason: str | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class HarnessInfo(BaseModel):
    id: str
    name: str
    version: str
    commit: str
    contract_version: str = CONTRACT_VERSION
    model: str
    model_actual: str | None = None
    implementation: str = "python"
    hermes_version: str | None = None


class CodeResponse(BaseModel):
    request_id: str
    case_id: str
    harness: HarnessInfo
    codes: list[CodeEntry] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0
    raw_output: str | None = None
    error: ErrorInfo | None = None
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    budget: BudgetReport = Field(default_factory=BudgetReport)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    harness: HarnessInfo
    contract_version: str = CONTRACT_VERSION
    uptime_s: float = 0.0
    redis_configured: bool = False


class ContractResponse(BaseModel):
    name: str = CONTRACT_NAME
    version: str = CONTRACT_VERSION
    request_schema: dict[str, Any]
    response_schema: dict[str, Any]


class JobStatus(BaseModel):
    job_id: str
    comparison_id: str
    slot: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELED"]
    attempt: int = 0
    error: str | None = None


def contract_payload() -> ContractResponse:
    return ContractResponse(
        request_schema=CodeRequest.model_json_schema(),
        response_schema=CodeResponse.model_json_schema(),
    )
