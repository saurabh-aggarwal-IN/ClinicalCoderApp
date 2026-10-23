from __future__ import annotations

import json
from dataclasses import replace

import pytest

from clinical_coder.contracts import CodeRequest, Usage
from clinical_coder.harnesses.langgraph_h3 import LangGraphH3Harness
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


PLAN = json.dumps(
    {
        "conditions": [
            {
                "condition": "Type 2 diabetes mellitus without complications",
                "status": "active",
                "evidence": "stable chronic T2DM without complications",
            }
        ],
        "candidate_codes": [{"code": "E11.9", "system": "ICD-10-CM", "why": "documented"}],
        "do_not_code": [{"code": "J44.1", "reason": "historical HCC only"}],
    }
)

CODES = json.dumps(
    {
        "codes": [
            {
                "code": "E11.9",
                "system": "ICD-10-CM",
                "snippet": "stable chronic T2DM without complications",
                "rationale": "Documented.",
            }
        ]
    }
)

CODES_FIXED = json.dumps(
    {
        "codes": [
            {
                "code": "E11.9",
                "system": "ICD-10-CM",
                "snippet": "stable chronic T2DM without complications",
                "rationale": "Documented and confirmed.",
            }
        ]
    }
)

VERIFIER_CLEAN = json.dumps(
    {
        "approved": True,
        "issues": [],
        "codes": json.loads(CODES)["codes"],
    }
)

VERIFIER_REJECT = json.dumps(
    {
        "approved": False,
        "issues": [
            {
                "code": "J44.1",
                "problem": "Historical HCC is not documented in the current note.",
                "action": "remove",
            }
        ],
        "codes": json.loads(CODES_FIXED)["codes"],
    }
)


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


@pytest.mark.asyncio
async def test_h3_verifier_approves(cfg, note, prompt) -> None:
    cfg = replace(cfg, server_max_llm_calls=8)
    service, gateway = make_service(cfg, [PLAN, CODES, VERIFIER_CLEAN])
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert [entry.code for entry in response.codes] == ["E11.9"]
    assert response.harness.id == "h3"
    assert len(gateway.calls) == 3
    nodes = [
        event.payload.get("node")
        for event in response.trace
        if event.type == "turn_start"
    ]
    assert nodes.count("verifier") == 1
    assert nodes.count("coder") == 1
    assert any(event.type == "history_read" for event in response.trace)


@pytest.mark.asyncio
async def test_h3_rejects_then_revises_once(cfg, note, prompt) -> None:
    cfg = replace(cfg, server_max_llm_calls=8)
    service, gateway = make_service(
        cfg,
        [PLAN, CODES, VERIFIER_REJECT, CODES_FIXED, VERIFIER_CLEAN],
    )
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert [entry.code for entry in response.codes] == ["E11.9"]
    assert len(gateway.calls) == 5
    starts = [
        event.payload.get("node")
        for event in response.trace
        if event.type == "turn_start"
    ]
    assert starts.count("coder") == 2
    assert starts.count("verifier") == 2
    verifier_turns = [
        event for event in response.trace
        if event.type == "turn_end" and event.payload.get("node") == "verifier"
    ]
    assert verifier_turns[0].payload["needs_revision"] is True
    assert verifier_turns[1].payload["needs_revision"] is False
    assert any("verifier rejected" in msg["content"].lower() for msg in gateway.calls[3])


@pytest.mark.asyncio
async def test_h3_malformed_verifier_can_be_repaired(cfg, note, prompt) -> None:
    cfg = replace(cfg, server_max_llm_calls=8)
    service, gateway = make_service(
        cfg,
        [PLAN, CODES, "not json", CODES_FIXED, VERIFIER_CLEAN],
    )
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert [entry.code for entry in response.codes] == ["E11.9"]
    assert len(gateway.calls) == 5
    verifier_turns = [
        event for event in response.trace
        if event.type == "turn_end" and event.payload.get("node") == "verifier"
    ]
    assert verifier_turns[0].payload["needs_revision"] is True


@pytest.mark.asyncio
async def test_h3_budget_error_is_structured(cfg, note, prompt) -> None:
    cfg = replace(cfg, server_max_llm_calls=8)
    service, _ = make_service(
        cfg,
        [PLAN, CODES, VERIFIER_CLEAN],
        fail_on_call=2,
    )
    response = await service.handle(request(note, prompt))
    assert response.error is not None
    assert response.error.code == "BUDGET_EXCEEDED"
