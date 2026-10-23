from __future__ import annotations

import json

import pytest

from clinical_coder.contracts import CodeRequest, Usage
from clinical_coder.harnesses.langgraph_h2 import LangGraphH2Harness
from clinical_coder.llm.client import LLMError, LLMResponse
from clinical_coder.service import HarnessService


class FakeGateway:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(messages)
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
                "history_support": "E11.9",
            },
            {
                "condition": "Chronic systolic heart failure",
                "status": "active",
                "evidence": "stable chronic systolic heart failure",
                "history_support": "I50.22",
            },
        ],
        "candidate_codes": [
            {"code": "E11.9", "system": "ICD-10-CM", "why": "documented"},
            {"code": "I50.22", "system": "ICD-10-CM", "why": "documented"},
        ],
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
            },
            {
                "code": "I50.22",
                "system": "ICD-10-CM",
                "snippet": "stable chronic systolic heart failure",
                "rationale": "Documented.",
            },
        ]
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
            "priorClaims": [{"diagnoses": ["I50.22"], "procedures": ["93000"]}],
        },
        request_id="h2-test",
    )


def make_service(cfg, contents: list[str]) -> tuple[HarnessService, FakeGateway]:
    gateway = FakeGateway(contents)
    harness = LangGraphH2Harness(cfg, gateway_factory=lambda _cfg, _budget: gateway)
    return HarnessService(harness, cfg), gateway


@pytest.mark.asyncio
async def test_h2_graph_runs_planner_then_coder(cfg, note, prompt) -> None:
    service, gateway = make_service(cfg, [PLAN, CODES])
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert [entry.code for entry in response.codes] == ["E11.9", "I50.22"]
    assert response.harness.id == "h2"
    assert len(gateway.calls) == 2
    assert any(event.type == "history_read" for event in response.trace)
    history_event = next(event for event in response.trace if event.type == "history_read")
    assert set(history_event.payload["codes"]) == {"E11.9", "I50.22", "J44.1", "93000"}
    assert any(event.type == "turn_start" and event.payload.get("node") == "planner" for event in response.trace)
    assert any(event.type == "turn_start" and event.payload.get("node") == "coder" for event in response.trace)
    assert not any(
        event.type == "turn_start" and event.payload.get("node") == "verifier"
        for event in response.trace
    )


@pytest.mark.asyncio
async def test_h2_prompts_include_history(cfg, note, prompt) -> None:
    service, gateway = make_service(cfg, [PLAN, CODES])
    await service.handle(request(note, prompt))
    planner_messages = gateway.calls[0]
    coder_messages = gateway.calls[1]
    assert "SYN-C01" in planner_messages[1]["content"]
    assert "J44.1" in planner_messages[1]["content"]
    assert "J44.1" in coder_messages[1]["content"]


@pytest.mark.asyncio
async def test_h2_malformed_coder_output_is_structured(cfg, note, prompt) -> None:
    service, _ = make_service(cfg, [PLAN, "not json"])
    response = await service.handle(request(note, prompt))

    assert response.error is not None
    assert response.error.code == "OUTPUT_FORMAT_INVALID"
    assert response.codes == []
    assert response.raw_output == "not json"


@pytest.mark.asyncio
async def test_h2_gateway_error_is_structured(cfg, note, prompt) -> None:
    class FailingGateway(FakeGateway):
        async def complete(self, messages, **kwargs):
            raise LLMError("CIRCUIT_OPEN", "breaker open", retryable=True)

    gateway = FailingGateway([])
    harness = LangGraphH2Harness(cfg, gateway_factory=lambda _cfg, _budget: gateway)
    service = HarnessService(harness, cfg)
    response = await service.handle(request(note, prompt))

    assert response.error is not None
    assert response.error.code == "CIRCUIT_OPEN"
    assert response.error.retryable is True
