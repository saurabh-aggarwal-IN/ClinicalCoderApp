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
