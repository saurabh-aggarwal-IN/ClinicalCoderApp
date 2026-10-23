from __future__ import annotations

import pytest
from pydantic import ValidationError

from clinical_coder.contracts import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    CodeEntry,
    CodeRequest,
    CodeResponse,
    HarnessInfo,
    contract_payload,
)


def test_contract_identity_is_frozen() -> None:
    assert CONTRACT_NAME == "clinical-coding-harness-v1"
    assert CONTRACT_VERSION == "1.0.0"


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CodeRequest.model_validate(
            {
                "case_id": "C01",
                "encounter_note": "note",
                "prompt": "prompt",
                "forbiddenCodes": ["X"],
            }
        )


def test_response_round_trip() -> None:
    response = CodeResponse(
        request_id="r1",
        case_id="C01",
        harness=HarnessInfo(
            id="h1",
            name="Direct Hermes",
            version="1.0.0",
            commit="abc",
            model="m",
        ),
        codes=[
            CodeEntry(
                code="E11.9",
                system="ICD-10-CM",
                snippet="type 2 diabetes mellitus",
                rationale="Documented.",
            )
        ],
    )
    payload = response.model_dump(mode="json")
    restored = CodeResponse.model_validate(payload)
    assert restored.codes[0].code == "E11.9"
    assert restored.disclaimer


def test_contract_payload_exposes_both_schemas() -> None:
    payload = contract_payload()
    assert payload.name == CONTRACT_NAME
    assert "properties" in payload.request_schema
    assert "codes" in payload.response_schema["properties"]
    assert "required" in payload.response_schema
    assert payload.request_schema["properties"]["budget"]["$ref"].startswith("#/$defs/")


def test_budget_spec_has_server_safe_bounds() -> None:
    with pytest.raises(ValidationError):
        CodeRequest.model_validate(
            {
                "case_id": "C01",
                "encounter_note": "note",
                "prompt": "prompt",
                "budget": {"max_llm_calls": 0},
            }
        )
