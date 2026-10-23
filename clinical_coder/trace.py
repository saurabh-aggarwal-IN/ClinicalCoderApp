from __future__ import annotations

import json
import time
from typing import Any

from clinical_coder.contracts import TraceEvent


def _jsonable(value: Any) -> Any:
    """Keep trace payloads JSON-safe without dropping useful structure."""
    return json.loads(json.dumps(value, default=str))


class TraceRecorder:
    def __init__(self) -> None:
        self._start = time.monotonic()
        self._events: list[TraceEvent] = []
        self._seq = 0

    def add(self, event_type: str, payload: dict[str, Any] | None = None) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(
            seq=self._seq,
            t_ms=max(0, int((time.monotonic() - self._start) * 1000)),
            type=event_type,  # type: ignore[arg-type]
            payload=_jsonable(payload or {}),
        )
        self._events.append(event)
        return event

    def message(self, text: str, **payload: Any) -> TraceEvent:
        return self.add("message", {"role": "assistant", "text": text, **payload})

    def tool_call(self, name: str, tool_input: dict[str, Any], **payload: Any) -> TraceEvent:
        return self.add(
            "tool_call", {"name": name, "input": tool_input, **payload}
        )

    def tool_result(self, name: str, output: Any, **payload: Any) -> TraceEvent:
        return self.add(
            "tool_result", {"name": name, "output": output, **payload}
        )

    def history_read(self, codes: list[str], source: str = "request") -> TraceEvent:
        return self.add(
            "history_read",
            {"source": source, "codes": codes, "code_count": len(codes)},
        )

    def usage(self, payload: dict[str, Any]) -> TraceEvent:
        return self.add("usage", payload)

    def error(self, code: str, message: str, **payload: Any) -> TraceEvent:
        return self.add("error", {"code": code, "message": message, **payload})

    def guardrail(self, report: Any) -> TraceEvent:
        return self.add("guardrail", report.model_dump(mode="json"))

    def budget(self, report: Any) -> TraceEvent:
        return self.add("budget", report.model_dump(mode="json"))

    def turn_start(self, **payload: Any) -> TraceEvent:
        return self.add("turn_start", payload)

    def turn_end(self, **payload: Any) -> TraceEvent:
        return self.add("turn_end", payload)

    def to_list(self) -> list[TraceEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
