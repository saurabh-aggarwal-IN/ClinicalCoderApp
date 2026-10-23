from __future__ import annotations

import json

import pytest

from clinical_coder.contracts import CodeRequest, Usage
from clinical_coder.harnesses.direct_hermes import DirectHermesHarness, build_h1_query
from clinical_coder.hermes.runner import HermesRunResult, HermesTimeout
from clinical_coder.service import HarnessService


class FakeRunner:
    def __init__(self, result: HermesRunResult | Exception) -> None:
        self.result = result
        self.kwargs = {}

    async def run(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def version(self) -> str:
        return "Hermes test 0.0.0"


def output_json() -> str:
    return json.dumps(
        {
            "codes": [
                {
                    "code": "E11.9",
                    "system": "ICD-10-CM",
                    "snippet": "stable chronic T2DM without complications",
                    "rationale": "The note assesses stable T2DM.",
                },
                {
                    "code": "I50.22",
                    "system": "ICD-10-CM",
                    "snippet": "stable chronic systolic heart failure",
                    "rationale": "The note assesses stable systolic CHF.",
                },
                {
                    "code": "93000",
                    "system": "CPT",
                    "snippet": "A 12-lead EKG was performed in the office today",
                    "rationale": "The EKG was performed.",
                },
            ]
        }
    )


def success(output: str | None = None) -> HermesRunResult:
    return HermesRunResult(
        raw_output=output if output is not None else output_json(),
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150, calls=1),
        exit_code=0,
        duration_ms=1234,
        events=[{"type": "system", "session_id": "s1"}],
        session_id="s1",
    )


def request(note: str, prompt: str, history=None) -> CodeRequest:
    return CodeRequest(
        case_id="C01",
        encounter_note=note,
        prompt=prompt,
        history=history,
        request_id="test-request",
    )


def make_service(cfg, result: HermesRunResult | Exception) -> tuple[HarnessService, FakeRunner]:
    runner = FakeRunner(result)
    harness = DirectHermesHarness(cfg, runner=runner)  # type: ignore[arg-type]
    return HarnessService(harness, cfg), runner


@pytest.mark.asyncio
async def test_h1_returns_contract_and_provenance(cfg, note, prompt) -> None:
    service, _ = make_service(cfg, success())
    response = await service.handle(request(note, prompt))

    assert response.error is None
    assert [code.code for code in response.codes] == ["E11.9", "I50.22", "93000"]
    assert response.usage.total_tokens == 150
    assert response.harness.id == "h1"
    assert response.harness.hermes_version == "Hermes test 0.0.0"
    assert response.guardrails.provenance_ok is True
    assert any(event.type == "turn_start" for event in response.trace)
    assert any(event.type == "usage" for event in response.trace)
    assert any(event.type == "turn_end" for event in response.trace)
    assert not any(event.type == "history_read" for event in response.trace)


@pytest.mark.asyncio
async def test_h1_receives_note_and_prompt_but_not_history(cfg, note, prompt) -> None:
    service, runner = make_service(cfg, success())
    history = {"patientId": "SYN-SECRET", "activeHccs": [{"hcc": "HCC 85"}]}
    response = await service.handle(request(note, prompt, history=history))

    assert response.error is None
    assert "SYN-SECRET" not in runner.kwargs["query"]
    assert "HCC 85" not in runner.kwargs["query"]
    assert note in runner.kwargs["query"]
    assert prompt in runner.kwargs["query"]
    assert any("does not receive" in warning for warning in response.warnings)


def test_query_builder_is_deterministic(note, prompt) -> None:
    built = build_h1_query(request(note, prompt))
    assert built.startswith(prompt)
    assert "<encounter_note>" in built
    assert note in built


@pytest.mark.asyncio
async def test_h1_malformed_output_is_structured_error(cfg, note, prompt) -> None:
    service, _ = make_service(cfg, success(output="I could not do that."))
    response = await service.handle(request(note, prompt))

    assert response.error is not None
    assert response.error.code == "OUTPUT_FORMAT_INVALID"
    assert response.codes == []
    assert response.raw_output == "I could not do that."


@pytest.mark.asyncio
async def test_h1_hermes_failure_is_structured(cfg, note, prompt) -> None:
    result = HermesRunResult(
        raw_output="",
        usage=Usage(),
        exit_code=1,
        duration_ms=10,
        error="provider unavailable",
    )
    service, _ = make_service(cfg, result)
    response = await service.handle(request(note, prompt))

    assert response.error is not None
    assert response.error.code == "HERMES_FAILED"
    assert "provider unavailable" in response.error.message


@pytest.mark.asyncio
async def test_h1_timeout_is_retryable(cfg, note, prompt) -> None:
    service, _ = make_service(cfg, HermesTimeout("too slow"))
    response = await service.handle(request(note, prompt))

    assert response.error is not None
    assert response.error.code == "HERMES_TIMEOUT"
    assert response.error.retryable is True


@pytest.mark.asyncio
async def test_ungrounded_output_warns_but_returns_codes(cfg, note) -> None:
    payload = json.loads(output_json())
    payload["codes"][0]["snippet"] = "a phrase that is not in the note"
    service, _ = make_service(cfg, success(json.dumps(payload)))
    response = await service.handle(request(note, "Assign codes."))

    assert response.error is None
    assert response.guardrails.provenance_ok is False
    assert "UNGROUNDED_CODE" in response.guardrails.output_flags
    assert any("provenance" in warning.lower() for warning in response.warnings)
