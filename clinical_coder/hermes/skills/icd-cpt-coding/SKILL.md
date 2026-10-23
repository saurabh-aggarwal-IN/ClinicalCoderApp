---
name: icd-cpt-coding
description: "Evidence-first ICD-10-CM and CPT coding for synthetic clinical notes."
version: 1.0.0
author: Clinical Coding Harness
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [clinical-coding, icd-10-cm, cpt, evidence]
---

# Evidence-First Clinical Coding

## When to use

Use when the task asks for ICD-10-CM diagnosis codes or CPT procedure codes for a synthetic encounter note.

## Rules

1. Code only what the current encounter note documents. History may add specificity to a condition that the current note already documents, but history alone never justifies a code.
2. Use the most specific code supported by the note. Prefer a combination code when it fully expresses the documented conditions.
3. Never recapture a historical HCC, prior claim diagnosis, or resolved condition unless the current note documents it as active.
4. Every code needs a short verbatim snippet from the encounter note. Do not paraphrase the snippet.
5. Do not assign E/M codes (99202-99499). Procedure codes are allowed only when the note documents the procedure as performed.
6. Return only the JSON object required by the task. No markdown fences, no commentary.
