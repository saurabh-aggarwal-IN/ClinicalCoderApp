"""Input and output guardrails for clinical coding calls.

The guards are deliberately deterministic and fail closed for obvious PHI,
prompt-injection, key leakage, and malformed output. They do not try to be a
complete PHI detector; the deployment boundary still owns that control.
"""

from __future__ import annotations

import json
import re
from typing import Any

from clinical_coder.contracts import (
    CodeEntry,
    CodeRequest,
    GuardrailReport,
    GuardrailViolation,
)
from clinical_coder.output import (
    ICD_RE,
    CPT_RE,
    is_em_code,
    normalize_whitespace,
    provenance_entry,
)
from clinical_coder.settings import Settings

PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("MRN", re.compile(r"\bMRN\b\s*[:#]?\s*\d{5,}\b", re.IGNORECASE)),
    ("DOB_LABEL", re.compile(r"\bDOB\b\s*[:#]?", re.IGNORECASE)),
]

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "IGNORE_INSTRUCTIONS",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions\b",
            re.IGNORECASE,
        ),
    ),
    (
        "REVEAL_PROMPT",
        re.compile(r"\b(?:reveal|print|show|repeat|leak)\s+(?:the\s+)?(?:system|developer)\s+prompt\b", re.IGNORECASE),
    ),
    (
        "ROLE_OVERRIDE",
        re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.IGNORECASE),
    ),
    ("JAILBREAK", re.compile(r"\bjailbreak\b", re.IGNORECASE)),
    (
        "SYSTEM_OVERRIDE",
        re.compile(r"\b(?:system|developer)\s+(?:message|instruction)s?\s*:", re.IGNORECASE),
    ),
]

KEY_LEAK_PATTERNS = [
    "expectedCodes",
    "forbiddenCodes",
    "coding_key.json",
    "expected/",
    "historySnippet",
]


def _report(
    *,
    violations: list[GuardrailViolation] | None = None,
    input_flags: list[str] | None = None,
    output_flags: list[str] | None = None,
    phi_detected: bool = False,
    prompt_injection_detected: bool = False,
    key_leakage_detected: bool = False,
    provenance_ok: bool | None = None,
) -> GuardrailReport:
    violations = violations or []
    return GuardrailReport(
        blocked=any(v.severity == "block" for v in violations),
        input_flags=input_flags or [],
        output_flags=output_flags or [],
        violations=violations,
        phi_detected=phi_detected,
        prompt_injection_detected=prompt_injection_detected,
        key_leakage_detected=key_leakage_detected,
        provenance_ok=provenance_ok,
    )


def check_input(request: CodeRequest, cfg: Settings) -> GuardrailReport:
    violations: list[GuardrailViolation] = []
    input_flags: list[str] = []

    if len(request.encounter_note) > cfg.max_note_chars:
        violations.append(
            GuardrailViolation(
                code="INPUT_TOO_LARGE",
                severity="block",
                message=f"Encounter note exceeds {cfg.max_note_chars} characters.",
                evidence={"field": "encounter_note", "chars": len(request.encounter_note)},
            )
        )
    if len(request.prompt) > cfg.max_prompt_chars:
        violations.append(
            GuardrailViolation(
                code="INPUT_TOO_LARGE",
                severity="block",
                message=f"Prompt exceeds {cfg.max_prompt_chars} characters.",
                evidence={"field": "prompt", "chars": len(request.prompt)},
            )
        )
    if request.history is not None:
        history_chars = len(json.dumps(request.history, default=str))
        if history_chars > cfg.max_history_chars:
            violations.append(
                GuardrailViolation(
                    code="INPUT_TOO_LARGE",
                    severity="block",
                    message=f"History fixture exceeds {cfg.max_history_chars} characters.",
                    evidence={"field": "history", "chars": history_chars},
                )
            )

    combined = "\n".join(
        [request.encounter_note, request.prompt, json.dumps(request.history or {}, default=str)]
    )

    phi_hits = [
        label for label, pattern in PHI_PATTERNS if pattern.search(combined)
    ]
    if phi_hits:
        severity = "warn" if cfg.allow_phi else "block"
        violations.append(
            GuardrailViolation(
                code="PHI_DETECTED",
                severity=severity,
                message=(
                    "Possible PHI detected. Set ALLOW_PHI=true only for an "
                    "approved, compliant deployment."
                ),
                evidence={"patterns": phi_hits},
            )
        )
        input_flags.append("PHI_DETECTED")

    injection_hits = [
        label for label, pattern in INJECTION_PATTERNS if pattern.search(combined)
    ]
    if injection_hits:
        severity = "warn" if cfg.allow_prompt_injection else "block"
        violations.append(
            GuardrailViolation(
                code="PROMPT_INJECTION_SUSPECTED",
                severity=severity,
                message="The input contains text that looks like a prompt-injection attempt.",
                evidence={"patterns": injection_hits},
            )
        )
        input_flags.append("PROMPT_INJECTION_SUSPECTED")

    key_hits = [pattern for pattern in KEY_LEAK_PATTERNS if pattern in combined]
    if key_hits:
        violations.append(
            GuardrailViolation(
                code="GRADING_KEY_LEAKAGE",
                severity="block",
                message="The request appears to contain grading-key material.",
                evidence={"patterns": key_hits},
            )
        )
        input_flags.append("GRADING_KEY_LEAKAGE")

    return _report(
        violations=violations,
        input_flags=input_flags,
        phi_detected=bool(phi_hits),
        prompt_injection_detected=bool(injection_hits),
        key_leakage_detected=bool(key_hits),
    )


def check_output(
    codes: list[CodeEntry],
    request: CodeRequest,
    cfg: Settings,
) -> GuardrailReport:
    violations: list[GuardrailViolation] = []
    output_flags: list[str] = []

    if len(codes) > cfg.default_max_codes:
        violations.append(
            GuardrailViolation(
                code="TOO_MANY_CODES",
                severity="block",
                message=f"Output has {len(codes)} codes; maximum is {cfg.default_max_codes}.",
            )
        )
        output_flags.append("TOO_MANY_CODES")

    seen: set[tuple[str, str]] = set()
    provenance: list[dict[str, Any]] = []
    all_provenance = True
    for entry in codes:
        key = (entry.system, entry.code)
        if key in seen:
            violations.append(
                GuardrailViolation(
                    code="DUPLICATE_CODE",
                    severity="block",
                    message=f"Duplicate code {entry.code}.",
                    evidence={"code": entry.code},
                )
            )
        seen.add(key)

        if entry.system == "ICD-10-CM" and not ICD_RE.match(entry.code):
            violations.append(
                GuardrailViolation(
                    code="INVALID_CODE_FORMAT",
                    severity="block",
                    message=f"Invalid ICD-10-CM code {entry.code}.",
                )
            )
        if entry.system == "CPT" and not CPT_RE.match(entry.code):
            violations.append(
                GuardrailViolation(
                    code="INVALID_CODE_FORMAT",
                    severity="block",
                    message=f"Invalid CPT code {entry.code}.",
                )
            )
        if is_em_code(entry.code):
            violations.append(
                GuardrailViolation(
                    code="EM_CODE_PRESENT",
                    severity="warn",
                    message="E/M code present; this benchmark ignores E/M assignment.",
                    evidence={"code": entry.code},
                )
            )
            output_flags.append("EM_CODE_PRESENT")

        ok, source, reason = provenance_entry(
            entry, request.encounter_note, request.history
        )
        all_provenance = all_provenance and ok
        provenance.append(
            {
                "code": entry.code,
                "ok": ok,
                "source": source,
                "reason": reason,
            }
        )
        if not ok:
            violations.append(
                GuardrailViolation(
                    code="UNGROUNDED_CODE",
                    severity="block" if cfg.strict_guardrails else "warn",
                    message=f"{entry.code} has no verbatim supporting snippet.",
                    evidence={"code": entry.code, "snippet": entry.snippet},
                )
            )
            output_flags.append("UNGROUNDED_CODE")

    combined_output = "\n".join(
        f"{entry.code} {entry.snippet} {entry.rationale}" for entry in codes
    )
    if any(pattern in combined_output for pattern in KEY_LEAK_PATTERNS):
        violations.append(
            GuardrailViolation(
                code="GRADING_KEY_LEAKAGE",
                severity="block",
                message="The output appears to contain grading-key material.",
            )
        )
        output_flags.append("GRADING_KEY_LEAKAGE")

    return _report(
        violations=violations,
        output_flags=output_flags,
        provenance_ok=all_provenance,
        key_leakage_detected="GRADING_KEY_LEAKAGE" in output_flags,
    )


def check_text_for_phi(text: str) -> list[str]:
    return [label for label, pattern in PHI_PATTERNS if pattern.search(text)]


def merge(a: GuardrailReport, b: GuardrailReport) -> GuardrailReport:
    a.extend(b)
    return a
