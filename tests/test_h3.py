from __future__ import annotations

import json
from dataclasses import replace

import pytest

from clinical_coder.contracts import CodeEntry, CodeRequest, Usage
from clinical_coder.harnesses.langgraph_h3 import LangGraphH3Harness
from clinical_coder.harnesses.multi_agent_contracts import (
    AssignedCPTCode,
    AssignedICD10Code,
    ClinicalExtractionInput,
    ClinicalExtractionOutput,
    CodeActionRequest,
    CodeVerificationDecision,
    CPTAgentInput,
    CPTAgentOutput,
    ExtractedEntity,
    ICD10AgentInput,
    ICD10AgentOutput,
    VerifierAgentOutput,
    VerifierInput,
)
from clinical_coder.llm.client import LLMError, LLMResponse
from clinical_coder.service import HarnessService


class FakeGateway:
    def __init__(self, contents: list[str], fail_on_call: int | None = None) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict]] = []
        self.fail_on_call = fail_on_call

    async def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise LLMError("BUDGET_EXCEEDED", "call budget exhausted")
        if not self.contents:
            raise LLMError("LLM_TEST_EXHAUSTED", "no more fake responses")
        content = self.contents.pop(0)
        return LLMResponse(
            content=content,
            model=kwargs.get("model", "mock/model"),
            model_actual=kwargs.get("model", "mock/model"),
            usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150, calls=1),
            attempts=1,
            retry_reasons=[],
            latency_ms=10,
            request_id=kwargs.get("request_id"),
        )

    async def aclose(self) -> None:
        return None


def request(note: str, prompt: str) -> CodeRequest:
    return CodeRequest(
        case_id="C01",
        encounter_note=note,
        prompt=prompt,
        history={
            "patientId": "SYN-C01",
            "activeHccs": [{"supportingCodes": ["E11.9", "J44.1"]}],
        },
        request_id="h3-test",
    )


def make_service(cfg, contents: list[str], **kwargs) -> tuple[HarnessService, FakeGateway]:
    gateway = FakeGateway(contents, **kwargs)
    harness = LangGraphH3Harness(cfg, gateway_factory=lambda _cfg, _budget: gateway)
    return HarnessService(harness, cfg), gateway


# Test Fixture Payloads
EXTRACTION_OUTPUT = json.dumps(
    {
        "entities": [
            {
                "term": "Type 2 diabetes mellitus without complications",
                "entity_type": "ICD-10",
                "snippet": "stable chronic T2DM without complications",
                "context": "active chronic condition",
            },
            {
                "term": "Venipuncture routine",
                "entity_type": "CPT",
                "snippet": "routine venipuncture performed for labs",
                "context": "outpatient blood draw",
            },
        ]
    }
)

ICD10_OUTPUT = json.dumps(
    {
        "codes": [
            {
                "code": "E11.9",
                "system": "ICD-10-CM",
                "term": "Type 2 diabetes mellitus without complications",
                "snippet": "stable chronic T2DM without complications",
                "rationale": "Documented active T2DM without acute complications.",
            }
        ]
    }
)

CPT_OUTPUT = json.dumps(
    {
        "codes": [
            {
                "code": "36415",
                "system": "CPT",
                "term": "Venipuncture routine",
                "snippet": "routine venipuncture performed for labs",
                "rationale": "Routine venipuncture for lab draw.",
            }
        ]
    }
)

VERIFIER_ALL_APPROVED = json.dumps(
    {
        "decisions": [
            {
                "code": "E11.9",
                "system": "ICD-10-CM",
                "action": "approve",
                "reason": "Active condition documented with verbatim snippet.",
            },
            {
                "code": "36415",
                "system": "CPT",
                "action": "approve",
                "reason": "Routine venipuncture documented as performed.",
            },
        ],
        "approved_codes": [
            {
                "code": "E11.9",
                "system": "ICD-10-CM",
                "snippet": "stable chronic T2DM without complications",
                "rationale": "Documented active T2DM without acute complications.",
            },
            {
                "code": "36415",
                "system": "CPT",
                "snippet": "routine venipuncture performed for labs",
                "rationale": "Routine venipuncture for lab draw.",
            },
        ],
    }
)


@pytest.mark.asyncio
async def test_h3_happy_path_all_approved(cfg, note, prompt) -> None:
    """Tests the full multi-agent flow where Extractor -> ICD-10 + CPT -> Verifier approves all."""
    cfg = replace(cfg, server_max_llm_calls=8)
    service, gateway = make_service(
        cfg,
        [EXTRACTION_OUTPUT, ICD10_OUTPUT, CPT_OUTPUT, VERIFIER_ALL_APPROVED],
    )
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert response.harness.id == "h3"
    codes = [entry.code for entry in response.codes]
    assert "E11.9" in codes
    assert "36415" in codes
    assert len(gateway.calls) == 4

    nodes = [
        event.payload.get("node")
        for event in response.trace
        if event.type == "turn_start" and event.payload.get("node") is not None
    ]
    assert nodes == ["extractor", "icd10_agent", "cpt_agent", "verifier"]


@pytest.mark.asyncio
async def test_h3_verifier_deletes_unsupported_code(cfg, note, prompt) -> None:
    """Tests that Verifier can take action 'delete' to discard an unsupported or hallucinated code."""
    cfg = replace(cfg, server_max_llm_calls=8)

    icd10_with_extra = json.dumps(
        {
            "codes": [
                {
                    "code": "E11.9",
                    "system": "ICD-10-CM",
                    "term": "Type 2 diabetes mellitus",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "Documented.",
                },
                {
                    "code": "J44.1",
                    "system": "ICD-10-CM",
                    "term": "COPD exacerbation",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "From historical HCC.",
                },
            ]
        }
    )

    verifier_deletes_j44 = json.dumps(
        {
            "decisions": [
                {
                    "code": "E11.9",
                    "system": "ICD-10-CM",
                    "action": "approve",
                    "reason": "Active documented condition.",
                },
                {
                    "code": "J44.1",
                    "system": "ICD-10-CM",
                    "action": "delete",
                    "reason": "Historical HCC not documented in the current note.",
                },
            ],
            "approved_codes": [
                {
                    "code": "E11.9",
                    "system": "ICD-10-CM",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "Documented.",
                }
            ],
        }
    )

    service, gateway = make_service(
        cfg,
        [EXTRACTION_OUTPUT, icd10_with_extra, CPT_OUTPUT, verifier_deletes_j44],
    )
    response = await service.handle(request(note, prompt))

    assert response.error is None
    codes = [entry.code for entry in response.codes]
    assert codes == ["E11.9"]
    assert "J44.1" not in codes


@pytest.mark.asyncio
async def test_h3_verifier_replace_action_triggers_revision(cfg, note, prompt) -> None:
    """Tests that when Verifier flags a code with action 'replace':
    1. Replacement request is sent back to the respective specialist agent with better context/term.
    2. Sub-agent produces the better replacement code.
    3. Verifier re-evaluates and approves the replacement code.
    """
    cfg = replace(cfg, server_max_llm_calls=10)

    verifier_requests_replace = json.dumps(
        {
            "decisions": [
                {
                    "code": "E11.9",
                    "system": "ICD-10-CM",
                    "action": "replace",
                    "reason": "Note specifies diabetic CKD stage 3, requires combination code E11.22.",
                    "better_context_or_term": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
                },
                {
                    "code": "36415",
                    "system": "CPT",
                    "action": "approve",
                    "reason": "Routine venipuncture documented.",
                },
            ],
            "approved_codes": [
                {
                    "code": "36415",
                    "system": "CPT",
                    "snippet": "routine venipuncture performed for labs",
                    "rationale": "Routine venipuncture for lab draw.",
                }
            ],
        }
    )

    icd10_revised = json.dumps(
        {
            "codes": [
                {
                    "code": "E11.22",
                    "system": "ICD-10-CM",
                    "term": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "Combination code for T2DM with CKD based on verifier feedback.",
                }
            ]
        }
    )

    verifier_final = json.dumps(
        {
            "decisions": [
                {
                    "code": "E11.22",
                    "system": "ICD-10-CM",
                    "action": "approve",
                    "reason": "Correct specific combination code.",
                },
                {
                    "code": "36415",
                    "system": "CPT",
                    "action": "approve",
                    "reason": "Approved procedure.",
                },
            ],
            "approved_codes": [
                {
                    "code": "E11.22",
                    "system": "ICD-10-CM",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "Combination code for T2DM with CKD.",
                },
                {
                    "code": "36415",
                    "system": "CPT",
                    "snippet": "routine venipuncture performed for labs",
                    "rationale": "Routine venipuncture for lab draw.",
                },
            ],
        }
    )

    service, gateway = make_service(
        cfg,
        [
            EXTRACTION_OUTPUT,
            ICD10_OUTPUT,
            CPT_OUTPUT,
            verifier_requests_replace,
            icd10_revised,
            verifier_final,
        ],
    )
    response = await service.handle(request(note, prompt))

    assert response.error is None
    codes = [entry.code for entry in response.codes]
    assert "E11.22" in codes
    assert "36415" in codes
    assert "E11.9" not in codes
    assert len(gateway.calls) == 6

    icd10_revision_call = gateway.calls[4]
    found_replacement_context = any(
        "diabetic chronic kidney disease" in msg["content"].lower()
        for msg in icd10_revision_call
    )
    assert found_replacement_context


def test_h3_pydantic_contracts():
    """Verify that all communication schemas enforce strict Pydantic model contracts."""
    ext_in = ClinicalExtractionInput(encounter_note="Patient has hypertension and asthma.")
    assert ext_in.encounter_note == "Patient has hypertension and asthma."

    entity = ExtractedEntity(
        term="Hypertension",
        entity_type="ICD-10",
        snippet="Patient has hypertension",
    )
    ext_out = ClinicalExtractionOutput(entities=[entity])
    assert len(ext_out.entities) == 1
    assert ext_out.entities[0].entity_type == "ICD-10"

    icd_in = ICD10AgentInput(
        encounter_note="Patient has hypertension.",
        icd10_terms=[entity],
    )
    assert len(icd_in.icd10_terms) == 1

    icd_code = AssignedICD10Code(
        code="I10",
        system="ICD-10-CM",
        term="Hypertension",
        snippet="Patient has hypertension",
        rationale="Essential hypertension documented.",
    )
    icd_out = ICD10AgentOutput(codes=[icd_code])
    assert icd_out.codes[0].code == "I10"
    code_entry = icd_out.codes[0].to_code_entry()
    assert isinstance(code_entry, CodeEntry)

    cpt_entity = ExtractedEntity(
        term="ECG",
        entity_type="CPT",
        snippet="12-lead ECG performed",
    )
    cpt_in = CPTAgentInput(
        encounter_note="12-lead ECG performed.",
        cpt_terms=[cpt_entity],
    )
    cpt_code = AssignedCPTCode(
        code="93000",
        system="CPT",
        term="ECG",
        snippet="12-lead ECG performed",
        rationale="12-lead electrocardiogram.",
    )
    cpt_out = CPTAgentOutput(codes=[cpt_code])
    assert cpt_out.codes[0].code == "93000"

    ver_in = VerifierInput(
        encounter_note="Patient note",
        icd10_output=icd_out,
        cpt_output=cpt_out,
    )
    assert len(ver_in.icd10_output.codes) == 1
    assert len(ver_in.cpt_output.codes) == 1

    decisions = [
        CodeVerificationDecision(
            code="I10",
            system="ICD-10-CM",
            action="approve",
            reason="Supported",
        ),
        CodeVerificationDecision(
            code="93000",
            system="CPT",
            action="replace",
            reason="Check modifier",
            better_context_or_term="ECG with interpretation",
        ),
    ]
    ver_out = VerifierAgentOutput(
        decisions=decisions,
        approved_codes=[code_entry],
    )
    assert ver_out.decisions[0].action == "approve"
    assert ver_out.decisions[1].action == "replace"
    assert ver_out.decisions[1].better_context_or_term == "ECG with interpretation"


@pytest.mark.asyncio
async def test_h3_budget_error(cfg, note, prompt) -> None:
    """Verifies that budget failures in H3 are cleanly caught and reported."""
    cfg = replace(cfg, server_max_llm_calls=8)
    service, _ = make_service(
        cfg,
        [EXTRACTION_OUTPUT, ICD10_OUTPUT, CPT_OUTPUT, VERIFIER_ALL_APPROVED],
        fail_on_call=2,
    )
    response = await service.handle(request(note, prompt))
    assert response.error is not None
    assert response.error.code == "BUDGET_EXCEEDED"
