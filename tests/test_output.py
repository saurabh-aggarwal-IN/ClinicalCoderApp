from __future__ import annotations

import pytest

from clinical_coder.output import (
    OutputFormatError,
    code_set,
    extract_json_object,
    normalize_code,
    parse_coding_output,
    provenance_entry,
)


def test_extract_json_from_fence() -> None:
    value = extract_json_object(
        'Here is the result:\n```json\n{"codes": []}\n```\nDone.'
    )
    assert value == {"codes": []}


def test_extract_json_with_surrounding_text() -> None:
    value = extract_json_object('prefix {"codes": []} suffix')
    assert value == {"codes": []}


def test_parse_normalizes_codes_and_whitespace() -> None:
    codes = parse_coding_output(
        """
        {
          "codes": [
            {"code": " e11.9 ", "system": "icd10cm",
             "snippet": "stable   chronic T2DM", "rationale": "Documented."},
            {"code": "93000", "system": "CPT",
             "snippet": "A 12-lead EKG was performed", "rationale": "Performed."}
          ]
        }
        """
    )
    assert codes[0].code == "E11.9"
    assert codes[0].system == "ICD-10-CM"
    assert codes[0].snippet == "stable chronic T2DM"
    assert code_set(codes) == {"E11.9", "93000"}


def test_duplicate_is_rejected() -> None:
    payload = {
        "codes": [
            {"code": "E11.9", "system": "ICD-10-CM", "snippet": "abc", "rationale": "x"},
            {"code": "E11.9", "system": "ICD-10-CM", "snippet": "def", "rationale": "y"},
        ]
    }
    with pytest.raises(OutputFormatError, match="Duplicate"):
        parse_coding_output(__import__("json").dumps(payload))


def test_invalid_icd_shape_is_rejected() -> None:
    with pytest.raises(OutputFormatError, match="valid ICD"):
        parse_coding_output(
            '{"codes":[{"code":"11.9","system":"ICD-10-CM","snippet":"abc","rationale":"x"}]}'
        )


def test_invalid_system_is_rejected() -> None:
    with pytest.raises(OutputFormatError, match="unsupported system"):
        parse_coding_output(
            '{"codes":[{"code":"E11.9","system":"SNOMED","snippet":"abc","rationale":"x"}]}'
        )


def test_missing_codes_array_is_rejected() -> None:
    with pytest.raises(OutputFormatError, match="codes array"):
        parse_coding_output('{"result": []}')


def test_provenance_requires_verbatim_snippet(note: str) -> None:
    entry = parse_coding_output(
        '{"codes":[{"code":"E11.9","system":"ICD-10-CM",'
        '"snippet":"stable chronic T2DM without complications","rationale":"x"}]}'
    )[0]
    ok, source, _ = provenance_entry(entry, note, None)
    assert ok is True
    assert source == "encounter_note"

    paraphrased = entry.model_copy(update={"snippet": "diabetes is stable"})
    ok, source, _ = provenance_entry(paraphrased, note, None)
    assert ok is False
    assert source is None


def test_normalize_code() -> None:
    assert normalize_code(" e11.9 ") == "E11.9"
    assert normalize_code("E119") == "E119"
