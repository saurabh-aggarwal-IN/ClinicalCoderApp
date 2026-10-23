from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


class CircuitOpenError(RuntimeError):
    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"Provider circuit is open for {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._state = "closed"
        self._open_until = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def before_call(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._state == "open":
                if now < self._open_until:
                    raise CircuitOpenError(self._open_until - now)
                self._state = "half_open"
                self._probe_in_flight = False
            if self._state == "half_open":
                if self._probe_in_flight:
                    raise CircuitOpenError(self._cooldown_s)
                self._probe_in_flight = True

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._state = "closed"
            self._probe_in_flight = False
            self._open_until = 0.0

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            if self._state == "half_open" or self._failures >= self._threshold:
                self._state = "open"
                self._open_until = time.monotonic() + self._cooldown_s

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._state,
            "failures": self._failures,
            "open_until_monotonic": self._open_until,
        }


class AsyncRateLimiter:
    """Concurrency + requests-per-minute limiter for one provider process."""

    def __init__(self, max_concurrency: int, rpm: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self._semaphore.acquire()
        if self._interval > 0:
            async with self._lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                wait = max(0.0, self._next_allowed - now)
                self._next_allowed = max(now, self._next_allowed) + self._interval
            if wait > 0:
                await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *_: object) -> None:
        self._semaphore.release()


@dataclass
class ProviderRuntime:
    breaker: CircuitBreaker
    limiter: AsyncRateLimiter


_runtimes: dict[str, ProviderRuntime] = {}


def get_runtime(base_url: str, max_concurrency: int, rpm: int) -> ProviderRuntime:
    key = base_url.rstrip("/")
    runtime = _runtimes.get(key)
    if runtime is None:
        runtime = ProviderRuntime(
            breaker=CircuitBreaker(),
            limiter=AsyncRateLimiter(max_concurrency=max_concurrency, rpm=rpm),
        )
        _runtimes[key] = runtime
    return runtime


def reset_runtimes() -> None:
    _runtimes.clear()
