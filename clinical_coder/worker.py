from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import time
from typing import Any

from clinical_coder.contracts import CodeRequest
from clinical_coder.registry import build_harness
from clinical_coder.service import HarnessService
from clinical_coder.settings import settings

logger = logging.getLogger("clinical_coder.worker")


class RedisWorker:
    """Consume coding jobs from a Redis Stream and publish contract responses."""

    def __init__(self) -> None:
        self.cfg = settings()
        self.service = HarnessService(build_harness(self.cfg), self.cfg)
        self.consumer = (
            f"{self.cfg.harness_slot}-{socket.gethostname()}-{os.getpid()}"
        )
        self._stop = asyncio.Event()
        self._redis: Any = None

    async def run(self) -> None:
        from redis.asyncio import Redis
        from redis.exceptions import ResponseError

        self._redis = Redis.from_url(self.cfg.redis_url, decode_responses=True)
        try:
            try:
                await self._redis.xgroup_create(
                    self.cfg.stream_jobs,
                    self.cfg.consumer_group,
                    id="0",
                    mkstream=True,
                )
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
            logger.info(
                "worker ready slot=%s stream=%s group=%s consumer=%s",
                self.cfg.harness_slot,
                self.cfg.stream_jobs,
                self.cfg.consumer_group,
                self.consumer,
            )
            heartbeat = asyncio.create_task(self._heartbeat())
            try:
                await self._loop()
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
        finally:
            await self._redis.aclose()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self._reclaim()
            try:
                batches = await self._redis.xreadgroup(
                    self.cfg.consumer_group,
                    self.consumer,
                    {self.cfg.stream_jobs: ">"},
                    count=1,
                    block=5_000,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient Redis errors
                logger.warning("xreadgroup failed: %s", type(exc).__name__)
                await asyncio.sleep(2)
                continue
            for _stream, messages in batches or []:
                for message_id, fields in messages:
                    await self._handle(message_id, fields)

    async def _reclaim(self) -> None:
        try:
            _cursor, messages, _deleted = await self._redis.xautoclaim(
                self.cfg.stream_jobs,
                self.cfg.consumer_group,
                self.consumer,
                min_idle_time=60_000,
                start_id="0-0",
                count=5,
            )
        except Exception:  # noqa: BLE001 - older Redis or empty group
            return
        for message_id, fields in messages or []:
            if fields:
                logger.info("reclaimed job message=%s", message_id)
                await self._handle(message_id, fields)

    async def _handle(self, message_id: str, fields: dict[str, Any]) -> None:
        raw = fields.get("job")
        job: dict[str, Any] = {}
        try:
            job = json.loads(raw) if raw else {}
            request_payload = job.get("request") or {}
            if not isinstance(request_payload, dict):
                raise ValueError("job.request must be an object")
            request_payload.setdefault("request_id", job.get("job_id"))
            if job.get("model") and not request_payload.get("model"):
                request_payload["model"] = job["model"]
            request = CodeRequest.model_validate(request_payload)
            response = await self.service.handle(request)
            envelope = {
                "job_id": job.get("job_id") or message_id,
                "comparison_id": job.get("comparison_id") or self.cfg.comparison_id,
                "slot": job.get("slot") or self.cfg.harness_slot,
                "case_id": request.case_id,
                "trial": job.get("trial", 0),
                "model": request.model or self.cfg.llm_model,
                "response": response.model_dump(mode="json"),
            }
        except Exception as exc:  # noqa: BLE001 - never poison the consumer group
            logger.exception("job failed message=%s", message_id)
            envelope = {
                "job_id": job.get("job_id") or message_id,
                "comparison_id": job.get("comparison_id") or self.cfg.comparison_id,
                "slot": job.get("slot") or self.cfg.harness_slot,
                "case_id": job.get("case_id") or "unknown",
                "trial": job.get("trial", 0),
                "model": job.get("model") or self.cfg.llm_model,
                "error": {
                    "code": "WORKER_ERROR",
                    "message": f"{type(exc).__name__}: {exc}"[:2_000],
                    "retryable": False,
                },
            }

        await self._redis.xadd(
            self.cfg.stream_results,
            {"result": json.dumps(envelope, separators=(",", ":"))},
            maxlen=100_000,
            approximate=True,
        )
        await self._redis.xack(
            self.cfg.stream_jobs, self.cfg.consumer_group, message_id
        )
        await self._redis.hset(
            f"cc:job:{envelope['job_id']}",
            mapping={
                "comparison_id": envelope["comparison_id"],
                "slot": envelope["slot"],
                "case_id": envelope["case_id"],
                "status": "FAILED" if envelope.get("error") else "COMPLETED",
                "updated_at": str(int(time.time())),
            },
        )
        await self._redis.expire(f"cc:job:{envelope['job_id']}", 86_400)
        logger.info(
            "job done job=%s case=%s status=%s",
            envelope["job_id"],
            envelope["case_id"],
            "FAILED" if envelope.get("error") else "COMPLETED",
        )

    async def _heartbeat(self) -> None:
        key = f"cc:worker:{self.cfg.harness_slot}:{self.consumer}"
        while not self._stop.is_set():
            try:
                await self._redis.hset(
                    key,
                    mapping={
                        "slot": self.cfg.harness_slot,
                        "consumer": self.consumer,
                        "comparison_id": self.cfg.comparison_id,
                        "pid": os.getpid(),
                    },
                )
                await self._redis.expire(key, self.cfg.worker_heartbeat_s * 4)
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(self.cfg.worker_heartbeat_s)

    def stop(self) -> None:
        self._stop.set()


async def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    worker = RedisWorker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.stop)
        except NotImplementedError:
            pass
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
