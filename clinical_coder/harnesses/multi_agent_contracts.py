from __future__ import annotations

import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from clinical_coder.contracts import CodeEntry
from clinical_coder.output import normalize_code, normalize_system, normalize_whitespace


class ClinicalExtractionInput(BaseModel):
    """Pydantic input provided to the Clinical Extraction Agent."""

    model_config = ConfigDict(extra="ignore")

    encounter_note: str = Field(min_length=1, description="Verbatim synthetic encounter note")
    prompt: str = Field(default="", description="Task prompt / instructions")
    case_id: str = Field(default="", description="Identifier for this clinical encounter case")
    history: dict[str, Any] | None = Field(default=None, description="Patient history fixture if available")


class ExtractedEntity(BaseModel):
    """Clinically relevant term and its entity type extracted from the encounter note."""

    model_config = ConfigDict(extra="ignore")

    term: str = Field(min_length=1, description="Clinically relevant diagnosis, condition, symptom, or procedure term")
    entity_type: Literal["ICD-10", "CPT"] = Field(
        description="Entity type classification: ICD-10 for conditions/diagnoses/symptoms, CPT for procedures/services"
    )
    snippet: str = Field(min_length=1, description="Verbatim text snippet from the encounter note supporting this term")
    context: str = Field(default="", description="Clinical context such as active status, anatomical location, or severity")

    @field_validator("snippet")
    @classmethod
    def _normalize_snippet(cls, value: str) -> str:
        return normalize_whitespace(value)

    @field_validator("entity_type", mode="before")
    @classmethod
    def _normalize_entity_type(cls, value: Any) -> str:
        raw = str(value).upper().strip()
        if "CPT" in raw:
            return "CPT"
        return "ICD-10"


class ClinicalExtractionOutput(BaseModel):
    """Pydantic output produced by the Clinical Extraction Agent."""

    model_config = ConfigDict(extra="ignore")

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="List of extracted clinically relevant terms with their entity type and supporting evidence",
    )


class CodeActionRequest(BaseModel):
    """Replacement request passed from the Verifier back to specialist sub-agents."""

    model_config = ConfigDict(extra="ignore")

    original_code: str = Field(description="The code evaluated by the verifier")
    system: Literal["ICD-10-CM", "CPT"] = Field(description="Coding system")
    action: Literal["replace", "delete", "approve"] = Field(description="Action requested by the verifier")
    reason: str = Field(default="", description="Explanation of why this code requires replacement or modification")
    better_context_or_term: str = Field(
        default="",
        description="Better clinical context or refined clinically relevant term supplied by the verifier",
    )
    suggested_replacement: str | None = Field(
        default=None,
        description="Optional suggested replacement code or guidance",
    )


class ICD10AgentInput(BaseModel):
    """Pydantic input provided to the ICD-10 specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    encounter_note: str = Field(description="The clinical encounter note")
    icd10_terms: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Filtered clinically relevant terms of entity_type ICD-10",
    )
    history: dict[str, Any] | None = Field(default=None, description="Patient history fixture")
    replacement_requests: list[CodeActionRequest] = Field(
        default_factory=list,
        description="Requests from verifier for better replacement codes with enhanced context",
    )


class AssignedICD10Code(BaseModel):
    """ICD-10 code assigned by the ICD-10 specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Assigned ICD-10-CM code")
    system: Literal["ICD-10-CM"] = "ICD-10-CM"
    term: str = Field(default="", description="The clinically relevant term this code maps to")
    snippet: str = Field(min_length=1, description="Verbatim snippet from the encounter note")
    rationale: str = Field(default="", description="Clinical coding rationale")

    @field_validator("code")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return normalize_code(value)

    @field_validator("snippet")
    @classmethod
    def _clean_snippet(cls, value: str) -> str:
        return normalize_whitespace(value)

    def to_code_entry(self) -> CodeEntry:
        return CodeEntry(
            code=self.code,
            system="ICD-10-CM",
            snippet=self.snippet,
            rationale=self.rationale or f"Documented active condition: {self.term or self.code}",
        )


class ICD10AgentOutput(BaseModel):
    """Pydantic output produced by the ICD-10 specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    codes: list[AssignedICD10Code] = Field(
        default_factory=list,
        description="Assigned ICD-10-CM codes for the extracted terms",
    )


class CPTAgentInput(BaseModel):
    """Pydantic input provided to the CPT specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    encounter_note: str = Field(description="The clinical encounter note")
    cpt_terms: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Filtered clinically relevant terms of entity_type CPT",
    )
    history: dict[str, Any] | None = Field(default=None, description="Patient history fixture")
    replacement_requests: list[CodeActionRequest] = Field(
        default_factory=list,
        description="Requests from verifier for better replacement codes with enhanced context",
    )


class AssignedCPTCode(BaseModel):
    """CPT code assigned by the CPT specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Assigned CPT code")
    system: Literal["CPT"] = "CPT"
    term: str = Field(default="", description="The clinically relevant term this code maps to")
    snippet: str = Field(min_length=1, description="Verbatim snippet from the encounter note")
    rationale: str = Field(default="", description="Clinical coding rationale")

    @field_validator("code")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return normalize_code(value)

    @field_validator("snippet")
    @classmethod
    def _clean_snippet(cls, value: str) -> str:
        return normalize_whitespace(value)

    def to_code_entry(self) -> CodeEntry:
        return CodeEntry(
            code=self.code,
            system="CPT",
            snippet=self.snippet,
            rationale=self.rationale or f"Documented procedure performed: {self.term or self.code}",
        )


class CPTAgentOutput(BaseModel):
    """Pydantic output produced by the CPT specialist sub-agent."""

    model_config = ConfigDict(extra="ignore")

    codes: list[AssignedCPTCode] = Field(
        default_factory=list,
        description="Assigned CPT codes for the extracted terms",
    )


class VerifierInput(BaseModel):
    """Pydantic input provided to the Verifier Agent."""

    model_config = ConfigDict(extra="ignore")

    encounter_note: str = Field(description="The clinical encounter note")
    history: dict[str, Any] | None = Field(default=None, description="Patient history fixture")
    icd10_output: ICD10AgentOutput = Field(default_factory=ICD10AgentOutput)
    cpt_output: CPTAgentOutput = Field(default_factory=CPTAgentOutput)
    revision: int = Field(default=0, description="Current revision iteration count")


class CodeVerificationDecision(BaseModel):
    """Per-code evaluation action produced by the Verifier Agent."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="The code evaluated")
    system: Literal["ICD-10-CM", "CPT"] = Field(description="Coding system")
    action: Literal["approve", "delete", "replace"] = Field(
        description="Possible actions: approve (keep code), delete (remove code), replace (request better replacement code with better context/term)",
    )
    reason: str = Field(default="", description="Clinical rationale for approving, deleting, or requesting replacement")
    better_context_or_term: str = Field(
        default="",
        description="When action is replace: better context or clinically relevant term provided for specialist agents 2a/2b",
    )
    suggested_replacement: str | None = Field(
        default=None,
        description="Optional suggested replacement code or guidance",
    )

    @field_validator("code")
    @classmethod
    def _clean_code(cls, value: str) -> str:
        return normalize_code(value)

    @field_validator("system", mode="before")
    @classmethod
    def _clean_system(cls, value: Any) -> str:
        norm = normalize_system(str(value))
        return norm or "ICD-10-CM"

    @field_validator("action", mode="before")
    @classmethod
    def _clean_action(cls, value: Any) -> str:
        raw = str(value).lower().strip()
        if "approve" in raw or "keep" in raw or "accept" in raw:
            return "approve"
        if "delete" in raw or "remove" in raw or "reject" in raw:
            return "delete"
        if "replace" in raw or "revise" in raw or "substitute" in raw:
            return "replace"
        return "approve"


class VerifierAgentOutput(BaseModel):
    """Pydantic output produced by the Verifier Agent."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[CodeVerificationDecision] = Field(
        default_factory=list,
        description="Individual action decision for each evaluated ICD-10 and CPT code",
    )
    approved_codes: list[CodeEntry] = Field(
        default_factory=list,
        description="Final approved codes passing verification",
    )
