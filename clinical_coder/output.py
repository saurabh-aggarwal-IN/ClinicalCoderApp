from __future__ import annotations

import json
import re
from typing import Any, Iterable

from pydantic import ValidationError

from clinical_coder.contracts import CodeEntry

ICD_RE = re.compile(r"^[A-Z][0-9][A-Z0-9](?:\.[A-Z0-9]{1,4})?$")
CPT_RE = re.compile(r"^(?:[0-9]{5}|[0-9]{4}[A-Z])$")
EM_RE = re.compile(r"^99[2-4][0-9]{2}$")


class OutputFormatError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def normalize_code(code: str) -> str:
    return re.sub(r"\s+", "", code.upper().strip()).rstrip(".")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def normalize_system(system: str) -> str | None:
    value = re.sub(r"[\s_]+", "-", system.upper().strip())
    if value in {"ICD-10-CM", "ICD10CM", "ICD-10", "ICD10"}:
        return "ICD-10-CM"
    if value in {"CPT", "CPT-4", "CPT4"}:
        return "CPT"
    return None


def is_em_code(code: str) -> bool:
    return bool(EM_RE.match(normalize_code(code)))


def extract_json_object(text: str) -> Any:
    candidates: list[str] = [text.strip()]
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        candidates.append(match.group(1).strip())
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise OutputFormatError(
        "No parseable JSON object in the model output",
        details={"preview": text[:500]},
    )


def parse_coding_output(text: str) -> list[CodeEntry]:
    raw = extract_json_object(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("codes"), list):
        raise OutputFormatError(
            "Output does not contain a top-level codes array",
            details={"preview": text[:500]},
        )

    entries: list[CodeEntry] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw["codes"]):
        if not isinstance(item, dict):
            raise OutputFormatError(f"codes[{index}] is not an object")
        normalized = dict(item)
        system = normalize_system(str(normalized.get("system", "")))
        if system is None:
            raise OutputFormatError(
                f"codes[{index}] has unsupported system {normalized.get('system')!r}"
            )
        normalized["system"] = system
        normalized["code"] = normalize_code(str(normalized.get("code", "")))
        normalized["snippet"] = normalize_whitespace(str(normalized.get("snippet", "")))
        normalized["rationale"] = str(normalized.get("rationale", "")).strip()
        try:
            entry = CodeEntry.model_validate(normalized)
        except ValidationError as exc:
            raise OutputFormatError(
                f"codes[{index}] failed schema validation",
                details={"issues": exc.errors(include_url=False)},
            ) from exc
        _validate_code_shape(entry, index)
        key = (entry.system, entry.code)
        if key in seen:
            raise OutputFormatError(f"Duplicate code entry {entry.code}")
        seen.add(key)
        entries.append(entry)

    if not entries:
        raise OutputFormatError("Output contains no codes")
    return entries


def _validate_code_shape(entry: CodeEntry, index: int) -> None:
    if entry.system == "ICD-10-CM" and not ICD_RE.match(entry.code):
        raise OutputFormatError(
            f"codes[{index}] is not a valid ICD-10-CM code: {entry.code!r}"
        )
    if entry.system == "CPT" and not CPT_RE.match(entry.code):
        raise OutputFormatError(
            f"codes[{index}] is not a valid CPT code: {entry.code!r}"
        )


def code_set(codes: Iterable[CodeEntry], *, ignore_em: bool = True) -> set[str]:
    result: set[str] = set()
    for entry in codes:
        if ignore_em and is_em_code(entry.code):
            continue
        result.add(entry.code)
    return result


def provenance_entry(
    entry: CodeEntry, note: str, history: Any | None
) -> tuple[bool, str | None, str]:
    note_norm = normalize_whitespace(note)
    snippet = normalize_whitespace(entry.snippet)
    if snippet and snippet in note_norm:
        return True, "encounter_note", "Verbatim note snippet."
    if history is not None:
        history_text = normalize_whitespace(json.dumps(history, default=str))
        if snippet and snippet in history_text:
            return True, "history", "Verbatim history snippet (requires key allowance)."
    return False, None, "Snippet is not present in the encounter note."
