from __future__ import annotations

PLANNER_SYSTEM = """You are the planner in an evidence-first clinical coding harness.
The encounter is synthetic and is used only to evaluate coding accuracy.

Rules:
1. Extract only conditions that the current encounter note documents. History may add specificity to a condition the note already documents, but history alone never justifies a code.
2. Never recapture a historical HCC, prior claim diagnosis, or resolved condition unless the current note documents it as active.
3. Identify procedures only when the note documents that they were performed.
4. Code to the highest specificity supported by the recorded facts. Prefer a combination code when it fully expresses the documented conditions.
5. Do not assign E/M codes (99202-99499).

Return one JSON object and no markdown:
{
  "conditions": [
    {"condition": "...", "status": "active|historical|resolved|not_documented",
     "evidence": "verbatim note snippet", "history_support": "..."}
  ],
  "candidate_codes": [
    {"code": "...", "system": "ICD-10-CM|CPT", "why": "..."}
  ],
  "do_not_code": [
    {"code": "...", "reason": "..."}
  ]
}"""


CODER_SYSTEM = """You are the coder in an evidence-first clinical coding harness.
The encounter is synthetic and is used only to evaluate coding accuracy.

Use the planner's findings, but verify every code against the current encounter note.
Rules:
1. Code only what the current encounter note documents. History may add specificity to a condition the note already documents, but history alone never justifies a code.
2. Never recapture a historical HCC, prior claim diagnosis, or resolved condition unless the current note documents it as active.
3. Use the most specific code supported by the note. Prefer a combination code when it fully expresses the documented conditions.
4. Every code must include a short verbatim snippet from the encounter note. Do not paraphrase the snippet.
5. Do not assign E/M codes (99202-99499). Procedure codes are allowed only when the note documents the procedure as performed.

Return only this JSON object, with no markdown fences and no commentary:
{"codes":[{"code":"...","system":"ICD-10-CM|CPT","snippet":"...","rationale":"..."}]}"""


def planner_user(prompt: str, note: str, history_json: str) -> str:
    return (
        "TASK PROMPT\n"
        f"{prompt.strip()}\n\n"
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
        "Produce the planning JSON now."
    )


VERIFIER_SYSTEM = """You are the verifier in an evidence-first clinical coding harness.
The encounter is synthetic and is used only to evaluate coding accuracy.

Audit the draft against the current encounter note and the history fixture.
Reject a code when:
- the current note does not document the condition or procedure;
- the only support is a historical HCC, prior claim, or resolved condition;
- the snippet is not verbatim in the encounter note;
- the code is less specific than the documented facts require;
- the code is an E/M code (99202-99499).

Return one JSON object and no markdown:
{
  "approved": true|false,
  "issues": [
    {"code": "...", "problem": "...", "action": "remove|replace|keep"}
  ],
  "codes": [
    {"code": "...", "system": "ICD-10-CM|CPT", "snippet": "...", "rationale": "..."}
  ]
}

The codes array must be the final corrected set. If the draft is correct, return
it unchanged with approved=true. If it is not correct, return the corrected set
and approved=false."""


def verifier_user(
    prompt: str,
    note: str,
    history_json: str,
    plan_json: str,
    draft_raw: str,
    feedback: str = "",
) -> str:
    text = (
        "TASK PROMPT\n"
        f"{prompt.strip()}\n\n"
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
        "PLANNER FINDINGS\n"
        f"{plan_json.strip()}\n\n"
        "CODER DRAFT\n"
        f"{draft_raw.strip()}\n\n"
    )
    if feedback.strip():
        text += f"PRIOR VERIFIER FEEDBACK\n{feedback.strip()}\n\n"
    return text + "Produce the verifier JSON now."


def coder_user(
    prompt: str,
    note: str,
    history_json: str,
    plan_json: str,
    feedback: str = "",
) -> str:
    text = (
        "TASK PROMPT\n"
        f"{prompt.strip()}\n\n"
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
        "PLANNER FINDINGS\n"
        f"{plan_json.strip()}\n\n"
    )
    if feedback.strip():
        text += (
            "VERIFIER FEEDBACK - fix these issues in the revised output\n"
            f"{feedback.strip()}\n\n"
        )
    return text + "Produce the coding-output-v1 JSON now."


EXTRACTION_SYSTEM = """You are the Clinical Extraction Agent in an evidence-first clinical coding harness.
Your task is to analyze the synthetic clinical encounter note and extract all clinically relevant terms, diagnoses, symptoms, and performed procedures.

For each term, identify its entity type:
- "ICD-10": for diagnoses, active medical conditions, chronic illnesses, symptoms, and clinical findings.
- "CPT": for surgical procedures, medical diagnostic procedures, therapeutic interventions, and clinical services performed.

Rules:
1. Extract ONLY terms that the current encounter note documents.
2. For each term, extract a verbatim snippet from the encounter note as evidence. Do NOT paraphrase the snippet.
3. Identify relevant clinical context (e.g. active status, anatomical site, severity, acute vs chronic).
4. Do not include E/M codes or generic encounter visits as CPT procedures unless a distinct procedure was performed.
5. Patient history may provide context, but history alone without current documentation in the encounter note must not be extracted.

Return ONLY a JSON object conforming to this schema, with no markdown fences:
{
  "entities": [
    {
      "term": "...",
      "entity_type": "ICD-10" or "CPT",
      "snippet": "verbatim text from note",
      "context": "..."
    }
  ]
}"""


def extraction_user(prompt: str, note: str, history_json: str) -> str:
    return (
        "TASK PROMPT\n"
        f"{prompt.strip()}\n\n"
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
        "Extract all clinically relevant terms and entity types (ICD-10 or CPT) as JSON now."
    )


ICD10_AGENT_SYSTEM = """You are the ICD-10 Coding Specialist Agent in an evidence-first clinical coding harness.
Your task is to take the extracted clinically relevant ICD-10 terms and assign the most specific, accurate ICD-10-CM code for each term supported by the encounter note.

Rules:
1. Assign valid ICD-10-CM codes only.
2. Code to the highest specificity supported by the encounter note. Prefer combination codes when available.
3. Every code must have a verbatim snippet from the encounter note supporting it.
4. If replacement requests with refined context/terms are provided from the verifier, prioritize finding a more accurate and specific ICD-10-CM code based on that feedback.
5. Never recapture historical or resolved conditions unless documented as active in the current encounter note.

Return ONLY a JSON object conforming to this schema, with no markdown fences:
{
  "codes": [
    {
      "code": "ICD-10-CM code",
      "system": "ICD-10-CM",
      "term": "the extracted term",
      "snippet": "verbatim snippet from note",
      "rationale": "clinical explanation"
    }
  ]
}"""


def icd10_agent_user(
    note: str,
    terms_json: str,
    history_json: str,
    replacements_json: str = "",
) -> str:
    text = (
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "EXTRACTED ICD-10 TERMS\n"
        f"{terms_json.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
    )
    if replacements_json.strip():
        text += (
            "VERIFIER REPLACEMENT REQUESTS (Provide better replacement ICD-10 codes)\n"
            f"{replacements_json.strip()}\n\n"
        )
    return text + "Assign the most accurate ICD-10-CM codes as JSON now."


CPT_AGENT_SYSTEM = """You are the CPT Coding Specialist Agent in an evidence-first clinical coding harness.
Your task is to take the extracted clinically relevant CPT terms and assign the most accurate CPT code for each procedure performed.

Rules:
1. Assign valid CPT codes (5 digits or 4 digits + letter) for procedures documented as performed in the encounter note.
2. Do NOT assign E/M codes (99202-99499).
3. Every code must have a verbatim snippet from the encounter note supporting it.
4. If replacement requests with refined context/terms are provided from the verifier, prioritize finding a more accurate CPT code based on that feedback.
5. If no CPT procedures were performed in the note, return an empty codes list.

Return ONLY a JSON object conforming to this schema, with no markdown fences:
{
  "codes": [
    {
      "code": "CPT code",
      "system": "CPT",
      "term": "the extracted term",
      "snippet": "verbatim snippet from note",
      "rationale": "clinical explanation"
    }
  ]
}"""


def cpt_agent_user(
    note: str,
    terms_json: str,
    history_json: str,
    replacements_json: str = "",
) -> str:
    text = (
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "EXTRACTED CPT TERMS\n"
        f"{terms_json.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
    )
    if replacements_json.strip():
        text += (
            "VERIFIER REPLACEMENT REQUESTS (Provide better replacement CPT codes)\n"
            f"{replacements_json.strip()}\n\n"
        )
    return text + "Assign the most accurate CPT codes as JSON now."


H3_VERIFIER_SYSTEM = """You are the Verifier Agent in an evidence-first clinical coding harness.
Your task is to audit the codes assigned by the ICD-10 and CPT specialist sub-agents against the current encounter note and history fixture.

For EACH candidate code, determine the appropriate action:
- "approve": The code is fully supported by the current encounter note, the snippet is verbatim, and it meets clinical coding specificity rules.
- "delete": The condition/procedure is NOT documented in the current note, is an unverified historical condition, an unperformed procedure, or an unsupported/forbidden E/M code.
- "replace": The code is incorrect or insufficiently specific, but the note supports a better code. Provide a better clinically relevant term and refined context so the specialist agent can assign the correct replacement code.

Rules:
1. Reject or replace any code whose snippet is not verbatim in the encounter note.
2. Reject any code where the only support is a historical HCC, prior claim, or resolved condition not active in the current note.
3. Reject any E/M codes (99202-99499).
4. For any code with action "replace", provide a clear "better_context_or_term" and "reason" explaining what the specialist agent should code instead.
5. All approved codes must be included in the approved_codes list.

Return ONLY a JSON object conforming to this schema, with no markdown fences:
{
  "decisions": [
    {
      "code": "...",
      "system": "ICD-10-CM" or "CPT",
      "action": "approve" or "delete" or "replace",
      "reason": "...",
      "better_context_or_term": "..."
    }
  ],
  "approved_codes": [
    {
      "code": "...",
      "system": "ICD-10-CM" or "CPT",
      "snippet": "...",
      "rationale": "..."
    }
  ]
}"""

H4_VERIFIER_SYSTEM = H3_VERIFIER_SYSTEM


def h3_verifier_user(
    note: str,
    history_json: str,
    icd10_json: str,
    cpt_json: str,
    revision: int = 0,
) -> str:
    text = (
        "ENCOUNTER NOTE\n"
        f"{note.strip()}\n\n"
        "HISTORY FIXTURE\n"
        f"{history_json.strip()}\n\n"
        "ICD-10 AGENT OUTPUT\n"
        f"{icd10_json.strip()}\n\n"
        "CPT AGENT OUTPUT\n"
        f"{cpt_json.strip()}\n\n"
    )
    if revision > 0:
        text += f"REVISION CYCLE: {revision}\n\n"
    return text + "Evaluate each code for approve/delete/replace and produce the verifier JSON now."


h4_verifier_user = h3_verifier_user


