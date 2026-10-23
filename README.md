# Clinical Coding Harness

Three clinical-coding harnesses behind one stable API, for a **synthetic-data
screening comparison**:

| Harness | Commit | Flow | History | Notes |
|---|---|---|---|---|
| **H1** | `h1` | real Hermes CLI | no | Toolsets disabled; skill preloaded |
| **H2** | `h2` | LangGraph Planner -> Coder | yes | same planner/coder as H3 |
| **H3** | `h3` | LangGraph Planner -> Coder -> Verifier | yes | adds one optional revision loop |

All three expose `clinical-coding-harness-v1` and return the same
`CodeResponse`. H1 and H2/H3 are **not** an apples-to-apples causal experiment:
H1 differs by orchestration, history context, and execution path. Report the
comparison as screening/descriptive, not as a causal effect.

**Synthetic data only. Not clinical advice, not a billing determination, and not
validated for patient care.**

## API

```http
GET  /health
GET  /v1/harness
GET  /v1/contract
POST /v1/code
```

Example:

```bash
curl -s http://127.0.0.1:8080/v1/code \
  -H 'content-type: application/json' \
  -d '{
    "case_id": "C01",
    "encounter_note": "58-year-old male ...",
    "prompt": "Assign ICD-10-CM and CPT codes. Return coding-output-v1 JSON.",
    "history": null,
    "budget": {"max_llm_calls": 6, "max_wall_seconds": 120}
  }'
```

The response always includes:

- `codes[]` in `coding-output-v1`
- `trace[]` (`turn_start`, `history_read`, `tool_call`, `usage`, `guardrail`,
  `budget`, `error`, `turn_end`)
- `usage` and `budget` (limits and consumed values)
- `guardrails` (input/output flags and violations)
- a structured `error` on failure instead of a dropped request

`POST /v1/code` is the direct path. In a comparison, the same harness image runs
`python -m clinical_coder.worker` and consumes Redis Stream jobs. The worker
returns the identical contract payload on the results stream.

## Request guards

The shared service applies these before any model or CLI call:

- note, prompt, and history size limits;
- PHI/PII heuristics (`ALLOW_PHI=false` blocks by default);
- prompt-injection heuristics (`ALLOW_PROMPT_INJECTION=false`);
- grading-key leakage detection (`expectedCodes`, `forbiddenCodes`,
  `coding_key.json`, `expected/`, `historySnippet`);
- per-request budget clamped to server ceilings.

After generation it validates:

- JSON shape and `coding-output-v1` fields;
- ICD-10-CM and CPT code formats;
- duplicates and maximum code count;
- E/M codes are flagged and ignored by grading;
- every snippet must be verbatim in the encounter note (or history), otherwise
  `UNGROUNDED_CODE` is raised. Non-strict mode records a warning; strict mode
  blocks the response.

## Budgets and retries

`BudgetSpec` sets per-request ceilings: LLM calls, input/output/total tokens,
wall time, and optional cost. The server clamps each value to its
`SERVER_MAX_*` setting, so a caller cannot raise the ceiling.

H2/H3 call the shared `LLMGateway`, which enforces:

- bounded exponential backoff with jitter and `Retry-After`;
- retries only for timeouts, network errors, 429, and 5xx;
- concurrency and RPM limits per provider;
- a circuit breaker that opens after repeated failures;
- token/call/wall/cost accounting, including conservative accounting for failed
  attempts;
- structured `LLM_*` errors.

H1 is a subprocess boundary, so it is bounded with:

- `platform_toolsets.cli: []` and `disabled_toolsets` in a fresh `HERMES_HOME`;
- `--max-turns`;
- `agent.run_budget_seconds` (the installed CLI rejects `--run-budget` for
  custom providers);
- an outer request timeout and process-group kill;
- post-run token/cost checks from the `stream-json` result event.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn clinical_coder.api:app --port 8080
```

On Linux/macOS use `.venv/bin/python`.

H1 needs the real Hermes CLI. The image is built on
`nousresearch/hermes-agent:latest`; do not substitute a lookalike:

```bash
docker build -t clinical-coder-h1:local --build-arg HARNESS_ID=h1 --build-arg GIT_COMMIT="$(git rev-parse HEAD)" .
docker run --rm -p 8080:8080 \
  -e LLM_BASE_URL=http://host:4010/v1 \
  -e LLM_API_KEY=... \
  -e LLM_MODEL=... \
  clinical-coder-h1:local
```

For the comparison stack, use `HarnessComparison/`: it starts Redis and
Postgres, checks out the three commits, builds one image per commit, dispatches
every case/model/trial through Redis Streams, persists results, and serves radar
payloads.

## Layout

```
clinical_coder/
  contracts.py        frozen request/response models
  service.py          shared guards, budget, errors, trace
  api.py              FastAPI surface
  worker.py           Redis Streams worker
  guardrails/         input/output guards
  llm/                gateway, budget, pricing, rate limit, breaker
  harnesses/          h1/h2/h3 implementations
  hermes/             real CLI runner + bundled skill
tests/                contract, guards, budget, H1, API tests
```

## Limits

- The clinical keys are author-validated, not certified-coder-reviewed.
- The synthetic set is small; per-case results are noisy.
- H1 receives no history fixture by design; history-dependent cases therefore
  measure a bundled difference, not a clean harness effect.
- Do not send PHI. Do not enable `ALLOW_PHI` unless the deployment has the
  surrounding compliance controls.
