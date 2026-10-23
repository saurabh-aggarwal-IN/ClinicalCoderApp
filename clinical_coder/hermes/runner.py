from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clinical_coder.contracts import Usage
from clinical_coder.settings import Settings

_SKILL_ROOT = Path(__file__).resolve().parent / "skills"


class HermesTimeout(RuntimeError):
    pass


class HermesUnavailable(RuntimeError):
    pass


@dataclass
class HermesRunResult:
    raw_output: str
    usage: Usage
    exit_code: int
    duration_ms: int
    events: list[dict[str, Any]] = field(default_factory=list)
    stderr: str = ""
    session_id: str | None = None
    error: str | None = None
    workdir: str | None = None


def _yaml_string(value: str) -> str:
    # JSON strings are valid YAML double-quoted scalars.
    return json.dumps(value, ensure_ascii=False)


def build_hermes_config(
    cfg: Settings,
    *,
    model: str,
    max_turns: int,
    run_budget_seconds: int,
) -> str:
    base_url = cfg.llm_base_url.rstrip("/")
    toolsets = [item.strip() for item in cfg.hermes_toolsets.split(",") if item.strip()]
    toolset_yaml = "[]"
    if toolsets:
        toolset_yaml = "[" + ", ".join(_yaml_string(item) for item in toolsets) + "]"

    # The CLI accepts --api-key, but passing secrets on a command line is
    # avoidable. The provider reads OPENAI_API_KEY from the isolated process
    # environment; the placeholder below is expanded by Hermes, not by us.
    return f"""platform_toolsets:
  cli: {toolset_yaml}

model:
  provider: {_yaml_string(cfg.hermes_model_provider)}
  default: {_yaml_string(model)}
  base_url: {_yaml_string(base_url)}
  api_key: "${{OPENAI_API_KEY}}"
  context_length: {cfg.hermes_context_length}

providers:
  {cfg.hermes_model_provider}:
    base_url: {_yaml_string(base_url)}
    api_key: "${{OPENAI_API_KEY}}"
    api_mode: chat_completions

agent:
  max_turns: {max_turns}
  run_budget_seconds: {max(1, run_budget_seconds)}
  disabled_toolsets:
    - web
    - browser
    - terminal
    - file
    - code_execution
    - vision
    - video
    - image_gen
    - video_gen
    - skills
    - todo
    - cronjob
    - tts
    - messaging
    - memory
    - delegation

auxiliary:
  title_generation:
    enabled: false

skills:
  creation_nudge_interval: 0
"""


class HermesRunner:
    """Run the real Hermes CLI in a fresh, tool-less HERMES_HOME."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._version: str | None = None
        self._version_checked_at: float = 0.0
        self._version_lock = asyncio.Lock()

    async def version(self) -> str | None:
        if self.cfg.hermes_version_cache_s <= 0:
            return await self._read_version()
        async with self._version_lock:
            age = time.monotonic() - self._version_checked_at
            if self._version_checked_at and age < self.cfg.hermes_version_cache_s:
                return self._version
            self._version = await self._read_version()
            self._version_checked_at = time.monotonic()
            return self._version

    async def _read_version(self) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cfg.hermes_bin,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except (OSError, asyncio.TimeoutError):
            return None
        text = (stdout or stderr).decode("utf-8", "replace").strip()
        return text.splitlines()[0][:200] if text else None

    def _skill_dir(self) -> Path:
        skill = _SKILL_ROOT / self.cfg.hermes_skill
        if not (skill / "SKILL.md").is_file():
            raise HermesUnavailable(
                f"Hermes skill {self.cfg.hermes_skill!r} is not bundled with the harness"
            )
        return skill

    async def run(
        self,
        *,
        query: str,
        encounter_note: str,
        model: str,
        max_turns: int,
        timeout_s: float,
    ) -> HermesRunResult:
        if not shutil.which(self.cfg.hermes_bin):
            raise HermesUnavailable(f"Hermes binary not found: {self.cfg.hermes_bin}")

        root = Path(
            tempfile.mkdtemp(prefix="cc-hermes-", dir=self._ensure_root())
        )
        home = root / "home"
        workspace = root / "workspace"
        home.mkdir(mode=0o700)
        (home / "skills").mkdir(mode=0o700)
        workspace.mkdir(mode=0o700)

        skill_dir = self._skill_dir()
        shutil.copytree(skill_dir, home / "skills" / self.cfg.hermes_skill)

        query_path = workspace / "query.txt"
        note_path = workspace / "encounter_note.txt"
        query_path.write_text(query, encoding="utf-8")
        note_path.write_text(encounter_note, encoding="utf-8")
        (home / "config.yaml").write_text(
            build_hermes_config(
                self.cfg,
                model=model,
                max_turns=max_turns,
                run_budget_seconds=max(1, int(timeout_s)),
            ),
            encoding="utf-8",
        )

        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(home),
            "HERMES_HOME": str(home),
            "OPENAI_API_KEY": self.cfg.llm_api_key,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "dumb",
            "NO_COLOR": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(home / "tmp"),
        }
        (home / "tmp").mkdir(mode=0o700)

        command = [
            self.cfg.hermes_bin,
            "chat",
            "--query-file",
            str(query_path),
            "--format",
            "stream-json",
            "--oneshot",
            "--max-turns",
            str(max_turns),
            "--skills",
            self.cfg.hermes_skill,
            "--in",
            str(workspace),
        ]

        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workspace),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            self._cleanup(root)
            raise HermesUnavailable(f"Could not start Hermes: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=max(1.0, timeout_s)
            )
        except asyncio.TimeoutError as exc:
            self._kill(proc)
            try:
                await asyncio.wait_for(proc.communicate(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            self._cleanup(root)
            raise HermesTimeout(
                f"Hermes exceeded the {timeout_s:.1f}s request budget"
            ) from exc

        duration_ms = int((time.monotonic() - started) * 1000)
        events, raw_output, usage, session_id, result_error, exit_code = _parse_stream(
            stdout.decode("utf-8", "replace")
        )
        if exit_code is None:
            exit_code = proc.returncode or 0
        error = result_error
        if error is None and exit_code != 0:
            error = f"Hermes exited with code {exit_code}"

        if not self.cfg.hermes_keep_workdir:
            self._cleanup(root)

        return HermesRunResult(
            raw_output=raw_output.strip(),
            usage=usage,
            exit_code=exit_code,
            duration_ms=duration_ms,
            events=events,
            stderr=stderr.decode("utf-8", "replace")[:20_000],
            session_id=session_id,
            error=error,
            workdir=str(root) if self.cfg.hermes_keep_workdir else None,
        )

    def _ensure_root(self) -> str:
        path = Path(self.cfg.hermes_home_root)
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            # Some read-only root filesystems only allow /tmp.
            fallback = Path(tempfile.gettempdir()) / "clinical-coder-hermes"
            fallback.mkdir(parents=True, exist_ok=True, mode=0o700)
            return str(fallback)
        return str(path)

    @staticmethod
    def _cleanup(root: Path) -> None:
        shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def _parse_stream(
    text: str,
) -> tuple[list[dict[str, Any]], str, Usage, str | None, str | None, int | None]:
    events: list[dict[str, Any]] = []
    chunks: list[str] = []
    usage = Usage()
    session_id: str | None = None
    result_error: str | None = None
    exit_code: int | None = None
    final_text: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        kind = event.get("type")
        if kind == "text" and isinstance(event.get("text"), str):
            chunks.append(event["text"])
        if kind == "system" and event.get("session_id"):
            session_id = str(event["session_id"])
        if kind == "result":
            session_id = str(event.get("session_id") or session_id or "") or None
            if isinstance(event.get("text"), str):
                final_text = event["text"]
            if event.get("error"):
                result_error = str(event["error"])
            if isinstance(event.get("exit_code"), int):
                exit_code = event["exit_code"]
            tokens = event.get("tokens") or {}
            if isinstance(tokens, dict):
                usage = Usage(
                    input_tokens=int(tokens.get("input") or 0),
                    output_tokens=int(tokens.get("output") or 0),
                    cache_read_tokens=int(tokens.get("cache_read") or 0),
                    cache_write_tokens=int(tokens.get("cache_write") or 0),
                    total_tokens=int(tokens.get("total") or 0),
                    calls=1,
                )
                usage.total_tokens = usage.total_tokens or (
                    usage.input_tokens
                    + usage.output_tokens
                    + usage.cache_read_tokens
                    + usage.cache_write_tokens
                )

    raw_output = final_text if final_text is not None else "".join(chunks)
    if usage.calls == 0 and raw_output:
        usage.calls = 1
    return events, raw_output, usage, session_id, result_error, exit_code
