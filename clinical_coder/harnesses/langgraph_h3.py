from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from clinical_coder.contracts import CodeEntry, CodeRequest, Usage
from clinical_coder.harnesses.base import (
    BaseHarness,
    HarnessExecutionError,
    HarnessResult,
)
from clinical_coder.harnesses.langgraph_common import (
    _merge_usage,
    _request_id,
    _warnings,
    extract_history_codes,
    history_json,
)
from clinical_coder.harnesses.multi_agent_contracts import (
    AssignedCPTCode,
    AssignedICD10Code,
    ClinicalExtractionInput,
    ClinicalExtractionOutput,
    CodeActionRequest,
    CPTAgentInput,
    CPTAgentOutput,
    ExtractedEntity,
    ICD10AgentInput,
    ICD10AgentOutput,
    VerifierAgentOutput,
    VerifierInput,
)
from clinical_coder.harnesses.prompts import (
    CPT_AGENT_SYSTEM,
    EXTRACTION_SYSTEM,
    H3_VERIFIER_SYSTEM,
    ICD10_AGENT_SYSTEM,
    cpt_agent_user,
    extraction_user,
    h3_verifier_user,
    icd10_agent_user,
)
from clinical_coder.llm.budget import BudgetTracker
from clinical_coder.llm.client import LLMGateway
from clinical_coder.output import OutputFormatError, extract_json_object, parse_coding_output
from clinical_coder.settings import Settings
from clinical_coder.trace import TraceRecorder


class H3GraphState(TypedDict, total=False):
    request: CodeRequest
    model: str
    trace: TraceRecorder
    gateway: LLMGateway
    budget: BudgetTracker
    history_codes: list[str]

    # Communication between agents via Pydantic structures
    extraction_input: ClinicalExtractionInput
    extraction_output: ClinicalExtractionOutput

    icd10_input: ICD10AgentInput
    icd10_output: ICD10AgentOutput

    cpt_input: CPTAgentInput
    cpt_output: CPTAgentOutput

    verifier_input: VerifierInput
    verifier_output: VerifierAgentOutput

    replacement_requests_icd10: list[CodeActionRequest]
    replacement_requests_cpt: list[CodeActionRequest]

    codes: list[CodeEntry]
    raw_output: str
    revisions: int
    needs_revision: bool

    usage: Usage
    warnings: list[str]
    model_actual: str | None
    error: str | None


async def extractor_node(state: H3GraphState) -> H3GraphState:
    """Step 1: Clinical Extraction Agent.
    Takes Pydantic structure with encounter note as input, extracts clinically relevant terms & entity types,
    and outputs a Pydantic ClinicalExtractionOutput.
    """
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]

    trace.turn_start(node="extractor", model=model)
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

    extraction_input = ClinicalExtractionInput(
        encounter_note=request.encounter_note,
        prompt=request.prompt,
        case_id=request.case_id,
        history=request.history,
    )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": extraction_user(
                extraction_input.prompt,
                extraction_input.encounter_note,
                history_json(request),
            ),
        },
    ]

    response = await gateway.complete(
        messages,
        model=model,
        temperature=min(0.2, request.temperature),
        max_tokens=min(3_000, budget.limits.max_output_tokens),
        request_id=_request_id(request, "extractor"),
    )
    usage = _merge_usage(state, response.usage)
    warnings = list(state.get("warnings") or [])

    try:
        raw_obj = extract_json_object(response.content)
        if isinstance(raw_obj, dict):
            extraction_output = ClinicalExtractionOutput.model_validate(raw_obj)
        elif isinstance(raw_obj, list):
            extraction_output = ClinicalExtractionOutput.model_validate({"entities": raw_obj})
        else:
            extraction_output = ClinicalExtractionOutput()
    except (OutputFormatError, Exception) as exc:
        extraction_output = ClinicalExtractionOutput()
        warnings = _warnings(
            state, f"Clinical Extraction agent output could not be parsed as Pydantic: {exc}"
        )

    trace.message(response.content[:20_000], role="extractor", node="extractor")
    trace.turn_end(
        node="extractor",
        model_actual=response.model_actual,
        entity_count=len(extraction_output.entities),
    )

    return {
        "history_codes": history_codes,
        "extraction_input": extraction_input,
        "extraction_output": extraction_output,
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


async def icd10_agent_node(state: H3GraphState) -> H3GraphState:
    """Step 2a: ICD-10 Specialist Agent.
    Filters ICD-10 terms from Pydantic output, assigns ICD-10 codes, and outputs Pydantic ICD10AgentOutput.
    Also handles replacement requests from the verifier.
    """
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]
    revisions = state.get("revisions", 0)

    trace.turn_start(node="icd10_agent", model=model, revision=revisions)

    extraction_output = state.get("extraction_output") or ClinicalExtractionOutput()
    icd10_terms = [
        entity for entity in extraction_output.entities if entity.entity_type == "ICD-10"
    ]
    replacement_requests = state.get("replacement_requests_icd10") or []

    # If this is a revision turn and no ICD-10 replacements were requested, preserve existing codes
    if revisions > 0 and not replacement_requests and state.get("icd10_output"):
        trace.turn_end(node="icd10_agent", code_count=len(state["icd10_output"].codes), skipped=True)
        return {"revisions": revisions}

    icd10_input = ICD10AgentInput(
        encounter_note=request.encounter_note,
        icd10_terms=icd10_terms,
        history=request.history,
        replacement_requests=replacement_requests,
    )

    terms_json = json.dumps(
        [term.model_dump() for term in icd10_input.icd10_terms], indent=2
    )
    replacements_json = (
        json.dumps([r.model_dump() for r in icd10_input.replacement_requests], indent=2)
        if icd10_input.replacement_requests
        else ""
    )

    messages = [
        {"role": "system", "content": ICD10_AGENT_SYSTEM},
        {
            "role": "user",
            "content": icd10_agent_user(
                icd10_input.encounter_note,
                terms_json,
                history_json(request),
                replacements_json,
            ),
        },
    ]

    response = await gateway.complete(
        messages,
        model=model,
        temperature=request.temperature,
        max_tokens=min(3_000, budget.limits.max_output_tokens),
        request_id=_request_id(
            request, f"icd10-revise-{revisions}" if replacement_requests else "icd10-agent"
        ),
    )
    usage = _merge_usage(state, response.usage)
    warnings = list(state.get("warnings") or [])

    try:
        raw_obj = extract_json_object(response.content)
        if isinstance(raw_obj, dict):
            icd10_output = ICD10AgentOutput.model_validate(raw_obj)
        elif isinstance(raw_obj, list):
            icd10_output = ICD10AgentOutput.model_validate({"codes": raw_obj})
        else:
            icd10_output = ICD10AgentOutput()
    except (OutputFormatError, Exception) as exc:
        icd10_output = ICD10AgentOutput()
        warnings = _warnings(
            state, f"ICD-10 agent output could not be parsed as Pydantic: {exc}"
        )

    # If this was a revision, update any existing ICD-10 output
    existing_output = state.get("icd10_output")
    if existing_output and replacement_requests:
        replaced_codes = {r.original_code for r in replacement_requests}
        merged_codes = [c for c in existing_output.codes if c.code not in replaced_codes]
        merged_codes.extend(icd10_output.codes)
        icd10_output = ICD10AgentOutput(codes=merged_codes)

    trace.message(response.content[:20_000], role="icd10_agent", node="icd10_agent")
    trace.turn_end(
        node="icd10_agent",
        code_count=len(icd10_output.codes),
        model_actual=response.model_actual,
    )

    return {
        "icd10_input": icd10_input,
        "icd10_output": icd10_output,
        "replacement_requests_icd10": [],
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


async def cpt_agent_node(state: H3GraphState) -> H3GraphState:
    """Step 2b: CPT Specialist Agent.
    Filters CPT terms from Pydantic output, assigns CPT codes, and outputs Pydantic CPTAgentOutput.
    Also handles replacement requests from the verifier.
    """
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]
    revisions = state.get("revisions", 0)

    trace.turn_start(node="cpt_agent", model=model, revision=revisions)

    extraction_output = state.get("extraction_output") or ClinicalExtractionOutput()
    cpt_terms = [
        entity for entity in extraction_output.entities if entity.entity_type == "CPT"
    ]
    replacement_requests = state.get("replacement_requests_cpt") or []

    # If this is a revision turn and no CPT replacements were requested, preserve existing codes
    if revisions > 0 and not replacement_requests and state.get("cpt_output"):
        trace.turn_end(node="cpt_agent", code_count=len(state["cpt_output"].codes), skipped=True)
        return {"revisions": revisions}

    # If there are no CPT terms extracted and no replacement requests, return empty CPT output
    if not cpt_terms and not replacement_requests:
        trace.turn_end(node="cpt_agent", code_count=0, empty=True)
        return {
            "cpt_input": CPTAgentInput(encounter_note=request.encounter_note, cpt_terms=[]),
            "cpt_output": CPTAgentOutput(codes=[]),
            "replacement_requests_cpt": [],
        }

    cpt_input = CPTAgentInput(
        encounter_note=request.encounter_note,
        cpt_terms=cpt_terms,
        history=request.history,
        replacement_requests=replacement_requests,
    )

    terms_json = json.dumps(
        [term.model_dump() for term in cpt_input.cpt_terms], indent=2
    )
    replacements_json = (
        json.dumps([r.model_dump() for r in cpt_input.replacement_requests], indent=2)
        if cpt_input.replacement_requests
        else ""
    )

    messages = [
        {"role": "system", "content": CPT_AGENT_SYSTEM},
        {
            "role": "user",
            "content": cpt_agent_user(
                cpt_input.encounter_note,
                terms_json,
                history_json(request),
                replacements_json,
            ),
        },
    ]

    response = await gateway.complete(
        messages,
        model=model,
        temperature=request.temperature,
        max_tokens=min(3_000, budget.limits.max_output_tokens),
        request_id=_request_id(
            request, f"cpt-revise-{revisions}" if replacement_requests else "cpt-agent"
        ),
    )
    usage = _merge_usage(state, response.usage)
    warnings = list(state.get("warnings") or [])

    try:
        raw_obj = extract_json_object(response.content)
        if isinstance(raw_obj, dict):
            cpt_output = CPTAgentOutput.model_validate(raw_obj)
        elif isinstance(raw_obj, list):
            cpt_output = CPTAgentOutput.model_validate({"codes": raw_obj})
        else:
            cpt_output = CPTAgentOutput()
    except (OutputFormatError, Exception) as exc:
        cpt_output = CPTAgentOutput()
        warnings = _warnings(
            state, f"CPT agent output could not be parsed as Pydantic: {exc}"
        )

    # If this was a revision, update any existing CPT output
    existing_output = state.get("cpt_output")
    if existing_output and replacement_requests:
        replaced_codes = {r.original_code for r in replacement_requests}
        merged_codes = [c for c in existing_output.codes if c.code not in replaced_codes]
        merged_codes.extend(cpt_output.codes)
        cpt_output = CPTAgentOutput(codes=merged_codes)

    trace.message(response.content[:20_000], role="cpt_agent", node="cpt_agent")
    trace.turn_end(
        node="cpt_agent",
        code_count=len(cpt_output.codes),
        model_actual=response.model_actual,
    )

    return {
        "cpt_input": cpt_input,
        "cpt_output": cpt_output,
        "replacement_requests_cpt": [],
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


async def verifier_node(state: H3GraphState) -> H3GraphState:
    """Step 3: Verifier Agent.
    Takes the output of both subagents (ICD-10 and CPT codes) as Pydantic structures.
    Evaluates actions for each code:
      a. Approve
      b. Delete
      c. Replace: requests better replacement code to agents 2a and 2b with better context/term.
    Outputs a Pydantic VerifierAgentOutput.
    """
    request = state["request"]
    trace = state["trace"]
    gateway = state["gateway"]
    budget = state["budget"]
    model = state["model"]
    revisions = state.get("revisions", 0)

    trace.turn_start(node="verifier", model=model, revision=revisions)

    icd10_output = state.get("icd10_output") or ICD10AgentOutput()
    cpt_output = state.get("cpt_output") or CPTAgentOutput()

    verifier_input = VerifierInput(
        encounter_note=request.encounter_note,
        history=request.history,
        icd10_output=icd10_output,
        cpt_output=cpt_output,
        revision=revisions,
    )

    messages = [
        {"role": "system", "content": H3_VERIFIER_SYSTEM},
        {
            "role": "user",
            "content": h3_verifier_user(
                verifier_input.encounter_note,
                history_json(request),
                verifier_input.icd10_output.model_dump_json(indent=2),
                verifier_input.cpt_output.model_dump_json(indent=2),
                revision=revisions,
            ),
        },
    ]

    response = await gateway.complete(
        messages,
        model=model,
        temperature=0.0,
        max_tokens=min(4_000, budget.limits.max_output_tokens),
        request_id=_request_id(request, f"verifier-{revisions + 1}"),
    )
    usage = _merge_usage(state, response.usage)
    warnings = list(state.get("warnings") or [])

    try:
        raw_obj = extract_json_object(response.content)
        if isinstance(raw_obj, dict):
            verifier_output = VerifierAgentOutput.model_validate(raw_obj)
        else:
            verifier_output = VerifierAgentOutput()
    except (OutputFormatError, Exception) as exc:
        verifier_output = VerifierAgentOutput()
        warnings = _warnings(
            state, f"Verifier agent output could not be parsed as Pydantic: {exc}"
        )

    # Process individual decisions for each code: approve, delete, replace
    replacement_requests_icd10: list[CodeActionRequest] = []
    replacement_requests_cpt: list[CodeActionRequest] = []
    approved_entries: list[CodeEntry] = []

    decisions_by_code: dict[str, str] = {
        d.code: d.action for d in verifier_output.decisions
    }

    # Evaluate actions
    for decision in verifier_output.decisions:
        if decision.action == "replace":
            req = CodeActionRequest(
                original_code=decision.code,
                system=decision.system,
                action="replace",
                reason=decision.reason,
                better_context_or_term=decision.better_context_or_term,
                suggested_replacement=decision.suggested_replacement,
            )
            if decision.system == "ICD-10-CM":
                replacement_requests_icd10.append(req)
            elif decision.system == "CPT":
                replacement_requests_cpt.append(req)

    # Collect approved codes
    if verifier_output.approved_codes:
        for entry in verifier_output.approved_codes:
            # Check if this code was deleted
            if decisions_by_code.get(entry.code) != "delete":
                approved_entries.append(entry)
    else:
        # Fallback to approved decisions from candidate subagent codes
        for code_item in icd10_output.codes:
            if decisions_by_code.get(code_item.code) == "approve":
                approved_entries.append(code_item.to_code_entry())
        for code_item in cpt_output.codes:
            if decisions_by_code.get(code_item.code) == "approve":
                approved_entries.append(code_item.to_code_entry())

    # Resilient fallback: If no codes explicitly approved but candidate codes exist, keep non-deleted codes
    if not approved_entries:
        for code_item in icd10_output.codes:
            if decisions_by_code.get(code_item.code) != "delete":
                approved_entries.append(code_item.to_code_entry())
        for code_item in cpt_output.codes:
            if decisions_by_code.get(code_item.code) != "delete":
                approved_entries.append(code_item.to_code_entry())

    has_replacements = bool(replacement_requests_icd10 or replacement_requests_cpt)
    can_revise = revisions < 1
    needs_revision = has_replacements and can_revise

    final_dict = {"codes": [entry.model_dump() for entry in approved_entries]}
    final_raw = json.dumps(final_dict, indent=2)

    trace.message(response.content[:20_000], role="verifier", node="verifier")
    trace.turn_end(
        node="verifier",
        approved_count=len(approved_entries),
        replacements_requested=len(replacement_requests_icd10) + len(replacement_requests_cpt),
        needs_revision=needs_revision,
        model_actual=response.model_actual,
    )

    return {
        "verifier_input": verifier_input,
        "verifier_output": verifier_output,
        "replacement_requests_icd10": replacement_requests_icd10,
        "replacement_requests_cpt": replacement_requests_cpt,
        "codes": approved_entries,
        "raw_output": final_raw,
        "revisions": revisions + (1 if needs_revision else 0),
        "needs_revision": needs_revision,
        "usage": usage,
        "warnings": warnings,
        "model_actual": response.model_actual,
    }


def verifier_route(state: H3GraphState) -> str:
    """Route back to subagents if verifier requested code replacement; else end."""
    return "icd10_agent" if state.get("needs_revision") else "__end__"


class LangGraphH3Harness(BaseHarness):
    """Preset Harness H3: Multi-Step Agent with Extractor -> ICD-10/CPT Specialists -> Verifier Loop.

    All inter-agent communication is structured in Pydantic models.
    """

    harness_id = "h3"
    harness_name = "LangGraph Multi-Agent Specialist (Extractor -> ICD-10/CPT -> Verifier Loop)"

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

    def build_graph(self):
        workflow = StateGraph(H3GraphState)
        workflow.add_node("extractor", extractor_node)
        workflow.add_node("icd10_agent", icd10_agent_node)
        workflow.add_node("cpt_agent", cpt_agent_node)
        workflow.add_node("verifier", verifier_node)

        workflow.add_edge(START, "extractor")
        workflow.add_edge("extractor", "icd10_agent")
        workflow.add_edge("icd10_agent", "cpt_agent")
        workflow.add_edge("cpt_agent", "verifier")

        workflow.add_conditional_edges(
            "verifier",
            verifier_route,
            {"icd10_agent": "icd10_agent", "__end__": END},
        )
        return workflow.compile()

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

        state: H3GraphState = {
            "request": request,
            "model": model,
            "trace": trace,
            "gateway": gateway,
            "budget": budget,
            "history_codes": [],
            "extraction_input": ClinicalExtractionInput(encounter_note=request.encounter_note),
            "extraction_output": ClinicalExtractionOutput(),
            "icd10_input": ICD10AgentInput(encounter_note=request.encounter_note),
            "icd10_output": ICD10AgentOutput(),
            "cpt_input": CPTAgentInput(encounter_note=request.encounter_note),
            "cpt_output": CPTAgentOutput(),
            "verifier_input": VerifierInput(encounter_note=request.encounter_note),
            "verifier_output": VerifierAgentOutput(),
            "replacement_requests_icd10": [],
            "replacement_requests_cpt": [],
            "codes": [],
            "raw_output": "",
            "revisions": 0,
            "needs_revision": False,
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
        raw_output = final_state.get("raw_output") or ""
        if not codes:
            try:
                codes = parse_coding_output(raw_output)
            except Exception:
                raise HarnessExecutionError(
                    "OUTPUT_FORMAT_INVALID",
                    "No approved clinical codes produced by the multi-agent harness",
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
