from __future__ import annotations

from fastapi.testclient import TestClient

from clinical_coder.api import create_app
from clinical_coder.contracts import CodeEntry, CodeRequest, Usage
from clinical_coder.harnesses.base import BaseHarness, HarnessResult
from clinical_coder.service import HarnessService
from clinical_coder.trace import TraceRecorder


class FakeHarness(BaseHarness):
    harness_id = "h2"
    harness_name = "Fake Graph"

    async def run(self, request: CodeRequest, budget, trace: TraceRecorder) -> HarnessResult:
        trace.history_read(["E11.9", "I50.22"])
        trace.turn_end(code_count=1)
        return HarnessResult(
            codes=[
                CodeEntry(
                    code="E11.9",
                    system="ICD-10-CM",
                    snippet="stable chronic T2DM without complications",
                    rationale="Documented.",
                )
            ],
            raw_output='{"codes":[]}',
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1),
            model_actual="mock/model",
        )


def make_client(cfg) -> TestClient:
    service = HarnessService(FakeHarness(cfg), cfg)
    return TestClient(create_app(cfg, service))


def test_health(cfg) -> None:
    client = make_client(cfg)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["harness"]["id"] == "h2"
    assert body["contract_version"] == "1.0.0"


def test_contract_endpoint(cfg) -> None:
    client = make_client(cfg)
    response = client.get("/v1/contract")
    assert response.status_code == 200
    assert response.json()["name"] == "clinical-coding-harness-v1"


def test_code_endpoint(cfg, note, prompt) -> None:
    client = make_client(cfg)
    response = client.post(
        "/v1/code",
        json={"case_id": "C01", "encounter_note": note, "prompt": prompt},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "C01"
    assert body["codes"][0]["code"] == "E11.9"
    assert body["harness"]["implementation"] == "python"
    assert any(event["type"] == "history_read" for event in body["trace"])


def test_extra_request_field_is_rejected(cfg, note, prompt) -> None:
    client = make_client(cfg)
    response = client.post(
        "/v1/code",
        json={
            "case_id": "C01",
            "encounter_note": note,
            "prompt": prompt,
            "forbiddenCodes": ["Z94.9"],
        },
    )
    assert response.status_code == 422
