from __future__ import annotations

from dataclasses import replace

import pytest

from clinical_coder.settings import Settings, reset_settings_cache, settings


@pytest.fixture()
def cfg(tmp_path) -> Settings:
    reset_settings_cache()
    base = settings()
    value = replace(
        base,
        harness_id="h1",
        harness_version="test",
        git_commit="test-commit",
        llm_base_url="http://mock.local/v1",
        llm_api_key="test-secret-key",
        llm_model="mock/clinical-coder",
        llm_max_retries=1,
        llm_timeout_s=5.0,
        server_max_llm_calls=4,
        server_max_input_tokens=20_000,
        server_max_output_tokens=2_000,
        server_max_total_tokens=25_000,
        server_max_wall_seconds=30.0,
        server_max_cost_usd=None,
        max_note_chars=20_000,
        max_history_chars=20_000,
        max_prompt_chars=10_000,
        max_raw_output_chars=20_000,
        hermes_home_root=str(tmp_path / "hermes"),
        hermes_keep_workdir=False,
        hermes_version_cache_s=0,
        llm_prices_file="",
    )
    yield value
    reset_settings_cache()


@pytest.fixture()
def note() -> str:
    return (
        "58-year-old male with type 2 diabetes mellitus and chronic systolic "
        "heart failure. Assessed as stable chronic T2DM without complications "
        "and stable chronic systolic heart failure. A 12-lead EKG was performed "
        "in the office today. Continue metformin and lisinopril."
    )


@pytest.fixture()
def prompt() -> str:
    return (
        "Assign accurate ICD-10-CM and CPT codes. Every code needs a verbatim "
        "snippet. Return coding-output-v1 JSON."
    )
