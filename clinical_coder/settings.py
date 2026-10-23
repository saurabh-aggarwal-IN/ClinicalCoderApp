from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from clinical_coder import __version__

DEFAULT_HARNESS_ID = "h1"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    harness_id: str
    harness_version: str
    git_commit: str
    implementation: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_s: float
    llm_max_retries: int
    llm_max_concurrency: int
    llm_rpm: int
    llm_json_mode: bool
    llm_prices_file: str

    server_max_llm_calls: int
    server_max_input_tokens: int
    server_max_output_tokens: int
    server_max_total_tokens: int
    server_max_wall_seconds: float
    server_max_cost_usd: float | None
    default_max_codes: int

    allow_phi: bool
    allow_prompt_injection: bool
    strict_guardrails: bool
    max_note_chars: int
    max_history_chars: int
    max_prompt_chars: int

    hermes_bin: str
    hermes_toolsets: str
    hermes_max_turns: int
    hermes_model_provider: str
    hermes_context_length: int
    hermes_home_root: str
    hermes_skill: str
    hermes_keep_workdir: bool
    request_timeout_s: float
    hermes_version_cache_s: int
    max_raw_output_chars: int

    redis_url: str
    comparison_id: str
    harness_slot: str
    stream_jobs: str
    stream_results: str
    consumer_group: str
    worker_concurrency: int
    worker_heartbeat_s: int

    disclaimer: str

    @property
    def llm_endpoint(self) -> str:
        base = self.llm_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"


@lru_cache(maxsize=1)
def settings() -> Settings:
    max_cost_raw = os.getenv("SERVER_MAX_COST_USD", "1.00")
    max_cost = None if max_cost_raw.strip().lower() in {"", "none", "null"} else float(max_cost_raw)
    return Settings(
        harness_id=os.getenv("HARNESS_ID", DEFAULT_HARNESS_ID),
        harness_version=os.getenv("HARNESS_VERSION", __version__),
        git_commit=os.getenv("GIT_COMMIT", "unknown"),
        implementation=os.getenv("HARNESS_IMPLEMENTATION", "python"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:4010/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "mock-key"),
        llm_model=os.getenv("LLM_MODEL", "mock/clinical-coder"),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 90.0, 1.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 3, 0),
        llm_max_concurrency=_int("LLM_MAX_CONCURRENCY", 4, 1),
        llm_rpm=_int("LLM_RPM", 60, 0),
        llm_json_mode=_bool("LLM_JSON_MODE", False),
        llm_prices_file=os.getenv("LLM_PRICES_FILE", ""),
        server_max_llm_calls=_int("SERVER_MAX_LLM_CALLS", 8, 1),
        server_max_input_tokens=_int("SERVER_MAX_INPUT_TOKENS", 60_000, 1_000),
        server_max_output_tokens=_int("SERVER_MAX_OUTPUT_TOKENS", 8_000, 256),
        server_max_total_tokens=_int("SERVER_MAX_TOTAL_TOKENS", 80_000, 1_000),
        server_max_wall_seconds=_float("SERVER_MAX_WALL_SECONDS", 180.0, 10.0),
        server_max_cost_usd=max_cost,
        default_max_codes=_int("MAX_CODES", 50, 1),
        allow_phi=_bool("ALLOW_PHI", False),
        allow_prompt_injection=_bool("ALLOW_PROMPT_INJECTION", False),
        strict_guardrails=_bool("GUARDRAIL_STRICT", False),
        max_note_chars=_int("MAX_NOTE_CHARS", 50_000, 1_000),
        max_history_chars=_int("MAX_HISTORY_CHARS", 100_000, 1_000),
        max_prompt_chars=_int("MAX_PROMPT_CHARS", 20_000, 500),
        hermes_bin=os.getenv("HERMES_BIN", "hermes"),
        hermes_toolsets=os.getenv("HERMES_TOOLSETS", ""),
        hermes_max_turns=_int("HERMES_MAX_TURNS", 16, 1),
        hermes_model_provider=os.getenv("HERMES_MODEL_PROVIDER", "harness"),
        hermes_context_length=_int("HERMES_CONTEXT_LENGTH", 200_000, 4_096),
        hermes_home_root=os.getenv("HERMES_HOME_ROOT", "/tmp/clinical-coder-hermes"),
        hermes_skill=os.getenv("HERMES_SKILL", "icd-cpt-coding"),
        hermes_keep_workdir=_bool("HERMES_KEEP_WORKDIR", False),
        request_timeout_s=_float("REQUEST_TIMEOUT_S", 240.0, 10.0),
        hermes_version_cache_s=_int("HERMES_VERSION_CACHE_S", 300, 0),
        max_raw_output_chars=_int("MAX_RAW_OUTPUT_CHARS", 100_000, 1_000),
        redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        comparison_id=os.getenv("COMPARISON_ID", "local"),
        harness_slot=os.getenv("HARNESS_SLOT", "h1"),
        stream_jobs=os.getenv("STREAM_JOBS", "cc:jobs:local:h1"),
        stream_results=os.getenv("STREAM_RESULTS", "cc:results:local"),
        consumer_group=os.getenv("CONSUMER_GROUP", "workers"),
        worker_concurrency=_int("WORKER_CONCURRENCY", 1, 1),
        worker_heartbeat_s=_int("WORKER_HEARTBEAT_S", 5, 1),
        disclaimer=(
            "Synthetic clinical coding evaluation only. Not clinical advice, "
            "not a billing determination, and not validated for patient care."
        ),
    )


def reset_settings_cache() -> None:
    settings.cache_clear()
