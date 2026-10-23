from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from clinical_coder.contracts import CodeEntry, CodeRequest, Usage
from clinical_coder.harnesses.base import (
    BaseHarness,
    HarnessExecutionError,
    HarnessResult,
)
from clinical_coder.harnesses.prompts import (
    CODER_SYSTEM,
    PLANNER_SYSTEM,
    coder_user,
    planner_user,
)
from clinical_coder.llm.budget import BudgetTracker
from clinical_coder.llm.client import LLMGateway
from clinical_coder.output import (
    ICD_RE,
    CPT_RE,
    OutputFormatError,
    extract_json_object,
    normalize_code,
    parse_coding_output,
)
from clinical_coder.settings import Settings
from clinical_coder.trace import TraceRecorder

_HISTORY_CODE_KEYS = {"diagnoses", "procedures", "supportingCodes", "code", "historyCodes"}


class GraphState(TypedDict, total=False):
    request: CodeRequest
    model: str
    trace: TraceRecorder
    gateway: LLMGateway
    budget: BudgetTracker
    history_codes: list[str]
    plan: dict[str, Any]
    plan_raw: str
    draft_raw: str
    final_raw: str
    codes: list[CodeEntry]
    parse_error: str | None
    feedback: str
    revisions: int
    issues: list[str]
    usage: Usage
    warnings: list[str]
    model_actual: str | None
    error: str | None


def history_json(request: CodeRequest) -> str:
    if request.history is None:
        return "{}"
    return json.dumps(request.history, indent=2, default=str)[:120_000]


def extract_history_codes(history: Any) -> list[str]:
    """Collect code-like values from explicit history code fields only."""
    found: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for item_key, item_value in value.items():
                visit(item_value, str(item_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str) or key not in _HISTORY_CODE_KEYS:
            return
        candidate = normalize_code(value)
        if ICD_RE.match(candidate) or CPT_RE.match(candidate):
            found.add(candidate)

    visit(history)
    return sorted(found)


def _merge_usage(state: GraphState, response_usage: Usage) -> Usage:
    usage = (state.get("usage") or Usage()).model_copy(deep=True)
    usage.add(
        input_tokens=response_usage.input_tokens,
        output_tokens=response_usage.output_tokens,
        cache_read_tokens=response_usage.cache_read_tokens,
        cache_write_tokens=response_usage.cache_write_tokens,
        reasoning_tokens=response_usage.reasoning_tokens,
        calls=max(1, response_usage.calls),
    )
    return usage


def _warnings(state: GraphState, message: str) -> list[str]:
    values = list(state.get("warnings") or [])
    if message not in values:
        values.append(message)
    return values


def _request_id(request: CodeRequest, suffix: str) -> str:
    base = request.request_id or f"{request.case_id}-{suffix}"
    return f"{base}:{suffix}"


async def planner_node(state: GraphState) -> GraphState:
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]

    trace.turn_start(node="planner", model=model)
    history_codes = extract_history_codes(request.history)
    trace.add(
        "history_read",
        {
            "source": "request",
            "codes": history_codes,
            "code_count": len(history_codes),
            "present": request.history is not None,
        },
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {
            "role": "user",
            "content": planner_user(
                request.prompt,
                request.encounter_note,
                history_json(request),
            ),
        },
    ]
    response = await gateway.complete(
        messages,
        model=model,
        temperature=min(0.2, request.temperature),
        max_tokens=min(2_500, budget.limits.max_output_tokens),
        request_id=_request_id(request, "planner"),
    )
    usage = _merge_usage(state, response.usage)
    warnings = list(state.get("warnings") or [])
    plan: dict[str, Any]
    try:
        parsed = extract_json_object(response.content)
        if isinstance(parsed, dict):
            plan = parsed
        else:
            plan = {"raw": parsed}
    except OutputFormatError:
        plan = {"raw": response.content}
        warnings = _warnings(
            state, "Planner output was not valid JSON; coder used the raw plan."
        )

    trace.message(response.content[:20_000], role="planner", node="planner")
    trace.turn_end(node="planner", model_actual=response.model_actual)
    return {
        "history_codes": history_codes,
        "plan": plan,
        "plan_raw": response.content,
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


async def coder_node(state: GraphState) -> GraphState:
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]
    feedback = state.get("feedback") or ""

    trace.turn_start(node="coder", model=model, revision=bool(feedback))
    messages = [
        {"role": "system", "content": CODER_SYSTEM},
        {
            "role": "user",
            "content": coder_user(
                request.prompt,
                request.encounter_note,
                history_json(request),
                json.dumps(state.get("plan") or {}, indent=2, default=str)[:20_000],
                feedback,
            ),
        },
    ]
    response = await gateway.complete(
        messages,
        model=model,
        temperature=request.temperature,
        max_tokens=min(4_000, budget.limits.max_output_tokens),
        request_id=_request_id(request, "coder-revise" if feedback else "coder"),
    )
    usage = _merge_usage(state, response.usage)

    warnings = list(state.get("warnings") or [])
    codes: list[CodeEntry] = []
    parse_error: str | None = None
    try:
        codes = parse_coding_output(response.content)
    except OutputFormatError as exc:
        parse_error = str(exc)
        warnings = _warnings(
            state,
            "Coder output did not satisfy coding-output-v1; "
            "the verifier or error path will handle it.",
        )

    trace.message(response.content[:20_000], role="coder", node="coder")
    trace.turn_end(node="coder", code_count=len(codes), parse_error=parse_error)
    return {
        "draft_raw": response.content,
        "codes": codes,
        "parse_error": parse_error,
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


class LangGraphHarness(BaseHarness):
    """Shared planner/coder execution for H2 and H3."""

    def __init__(
        self,
        cfg: Settings,
        gateway_factory: Any | None = None,
    ) -> None:
        super().__init__(cfg)
        self.gateway_factory = gateway_factory or (
            lambda config, budget: LLMGateway(config, budget)
        )
        self.graph = self.build_graph()

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError

    async def run(
        self,
        request: CodeRequest,
        budget: BudgetTracker,
        trace: TraceRecorder,
    ) -> HarnessResult:
        model = self.model_for(request)
        gateway = self.gateway_factory(self.cfg, budget)
        trace.turn_start(
            harness=self.harness_id,
            model=model,
            history_included=request.history is not None,
        )
        state: GraphState = {
            "request": request,
            "model": model,
            "trace": trace,
            "gateway": gateway,
            "budget": budget,
            "history_codes": [],
            "plan": {},
            "plan_raw": "",
            "draft_raw": "",
            "final_raw": "",
            "codes": [],
            "parse_error": None,
            "feedback": "",
            "revisions": 0,
            "issues": [],
            "usage": Usage(),
            "warnings": [],
            "model_actual": None,
            "error": None,
        }

        try:
            async with asyncio.timeout(max(1.0, budget.remaining_wall_s())):
                final_state = await self.graph.ainvoke(state)
        finally:
            await gateway.aclose()

        if final_state.get("error"):
            raise HarnessExecutionError("HARNESS_ERROR", str(final_state["error"]))

        codes = final_state.get("codes") or []
        raw_output = final_state.get("final_raw") or final_state.get("draft_raw") or ""
        if not codes:
            detail = final_state.get("parse_error") or "no codes were produced"
            raise HarnessExecutionError(
                "OUTPUT_FORMAT_INVALID",
                f"Harness output could not be parsed: {detail}",
                details={"raw_output": raw_output},
            )

        trace.turn_end(
            harness=self.harness_id,
            code_count=len(codes),
            revisions=final_state.get("revisions", 0),
        )
        return HarnessResult(
            codes=codes,
            raw_output=raw_output,
            usage=final_state.get("usage") or Usage(),
            model_actual=final_state.get("model_actual"),
            warnings=final_state.get("warnings") or [],
        )


def compile_graph(nodes: list[tuple[str, Any]], edges: list[tuple[str, str]]) -> Any:
    workflow = StateGraph(GraphState)
    for name, node in nodes:
        workflow.add_node(name, node)
    workflow.add_edge(START, nodes[0][0])
    for source, target in edges:
        if target == "__end__":
            workflow.add_edge(source, END)
        else:
            workflow.add_edge(source, target)
    return workflow.compile()
