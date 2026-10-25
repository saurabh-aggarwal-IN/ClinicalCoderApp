from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from clinical_coder.contracts import CodeRequest, CodeResponse, HarnessInfo
from clinical_coder.service import HarnessService
from clinical_coder.settings import Settings, reset_settings_cache
from clinical_coder.worker import RedisWorker


class FakeHarnessService(HarnessService):
    def __init__(self, delay: float = 0.05, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.in_flight = 0
        self.max_in_flight = 0
        self.handled_cases: list[str] = []

    async def handle(self, request: CodeRequest) -> CodeResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.handled_cases.append(request.case_id)
        try:
            if self.fail_on and request.case_id == self.fail_on:
                raise RuntimeError(f"Simulated failure on {request.case_id}")
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            return CodeResponse(
                request_id=request.request_id or f"req-{request.case_id}",
                case_id=request.case_id,
                harness=HarnessInfo(
                    id="test",
                    name="Test Harness",
                    version="1.0.0",
                    commit="abc1234",
                    model="test-model",
                ),
                codes=[],
            )
        finally:
            self.in_flight -= 1


class FakeRedis:
    def __init__(self, messages: list[tuple[str, dict[str, str]]]) -> None:
        self.messages = list(messages)
        self.results: list[dict[str, Any]] = []
        self.acked: list[tuple[str, str, str]] = []
        self.hashes: dict[str, dict[str, str]] = {}
        self.closed = False

    async def xgroup_create(self, stream: str, group: str, id: str = "0", mkstream: bool = True) -> None:
        pass

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        count: int = 1,
        block: int = 2000,
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        if not self.messages:
            # When empty, pause briefly to simulate block then return empty
            await asyncio.sleep(0.01)
            return []
        batch = self.messages[:count]
        self.messages = self.messages[count:]
        stream_name = next(iter(streams.keys()))
        return [(stream_name, batch)]

    async def xautoclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        start_id: str,
        count: int = 5,
    ) -> tuple[str, list[Any], list[Any]]:
        return ("0-0", [], [])

    async def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True) -> str:
        raw = fields.get("result", "{}")
        self.results.append(json.loads(raw))
        return f"{len(self.results)}-0"

    async def xack(self, stream: str, group: str, *ids: str) -> int:
        for mid in ids:
            self.acked.append((stream, group, mid))
        return len(ids)

    async def hset(self, name: str, mapping: dict[str, Any] | None = None, **kwargs: Any) -> int:
        self.hashes.setdefault(name, {}).update(mapping or kwargs)
        return len(mapping or kwargs)

    async def expire(self, name: str, seconds: int) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


def _make_test_job(case_id: str, job_id: str) -> tuple[str, dict[str, str]]:
    job = {
        "job_id": job_id,
        "comparison_id": "test-cmp",
        "slot": "h1",
        "case_id": case_id,
        "request": {
            "case_id": case_id,
            "encounter_note": f"Encounter note for {case_id}",
            "prompt": "Assign codes",
        },
    }
    return (f"msg-{job_id}", {"job": json.dumps(job)})


@pytest.mark.asyncio
async def test_worker_processes_4_jobs_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_CONCURRENCY", "4")
    reset_settings_cache()

    messages = [
        _make_test_job(f"case-0{i}", f"job-{i}") for i in range(1, 5)
    ]
    fake_redis = FakeRedis(messages)
    fake_service = FakeHarnessService(delay=0.05)

    worker = RedisWorker(service=fake_service, redis_client=fake_redis)
    assert worker.cfg.worker_concurrency == 4

    worker_task = asyncio.create_task(worker.run())

    # Wait until all 4 results are published
    start = time.monotonic()
    for _ in range(50):
        if len(fake_redis.results) == 4:
            break
        await asyncio.sleep(0.01)

    elapsed = time.monotonic() - start
    worker.stop()
    await worker_task

    assert len(fake_redis.results) == 4
    # All 4 cases were processed
    assert sorted(fake_service.handled_cases) == ["case-01", "case-02", "case-03", "case-04"]
    # Concurrency achieved: max_in_flight must be > 1 (ideally 4)
    assert fake_service.max_in_flight >= 3, f"Expected concurrent execution, max in flight was {fake_service.max_in_flight}"
    # All 4 jobs were acknowledged
    assert len(fake_redis.acked) == 4
    # With delay=0.05s each, running 4 concurrently should take much less than sequential
    assert elapsed < 0.35, f"Expected parallel speedup, took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_worker_concurrency_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_CONCURRENCY", "2")
    reset_settings_cache()

    messages = [
        _make_test_job(f"case-0{i}", f"job-{i}") for i in range(1, 5)
    ]
    fake_redis = FakeRedis(messages)
    fake_service = FakeHarnessService(delay=0.04)

    worker = RedisWorker(service=fake_service, redis_client=fake_redis)
    assert worker.cfg.worker_concurrency == 2

    worker_task = asyncio.create_task(worker.run())

    for _ in range(50):
        if len(fake_redis.results) == 4:
            break
        await asyncio.sleep(0.01)

    worker.stop()
    await worker_task

    assert len(fake_redis.results) == 4
    # Ceiling of 2 was strictly respected
    assert fake_service.max_in_flight <= 2


@pytest.mark.asyncio
async def test_worker_fault_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_CONCURRENCY", "4")
    reset_settings_cache()

    messages = [
        _make_test_job(f"case-0{i}", f"job-{i}") for i in range(1, 5)
    ]
    fake_redis = FakeRedis(messages)
    # Fail on case-02
    fake_service = FakeHarnessService(delay=0.02, fail_on="case-02")

    worker = RedisWorker(service=fake_service, redis_client=fake_redis)
    worker_task = asyncio.create_task(worker.run())

    for _ in range(50):
        if len(fake_redis.results) == 4:
            break
        await asyncio.sleep(0.01)

    worker.stop()
    await worker_task

    assert len(fake_redis.results) == 4
    # 3 succeeded, 1 has WORKER_ERROR
    failed = [r for r in fake_redis.results if r.get("error")]
    success = [r for r in fake_redis.results if not r.get("error")]
    assert len(failed) == 1
    assert failed[0]["case_id"] == "case-02"
    assert failed[0]["error"]["code"] == "WORKER_ERROR"
    assert len(success) == 3
    # All 4 were acked so poison pills don't hang stream
    assert len(fake_redis.acked) == 4
