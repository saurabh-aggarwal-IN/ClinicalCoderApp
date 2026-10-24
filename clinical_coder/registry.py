from __future__ import annotations

from clinical_coder.harnesses.base import BaseHarness
from clinical_coder.harnesses.direct_hermes import DirectHermesHarness
from clinical_coder.harnesses.langgraph_h2 import LangGraphH2Harness
from clinical_coder.harnesses.langgraph_h3 import LangGraphH3Harness
from clinical_coder.settings import Settings


def build_harness(cfg: Settings) -> BaseHarness:
    harness_id = (cfg.harness_id or "").strip().lower()
    if harness_id == "h1":
        return DirectHermesHarness(cfg)
    if harness_id == "h2":
        return LangGraphH2Harness(cfg)
    if harness_id == "h3":
        return LangGraphH3Harness(cfg)
    if not harness_id or harness_id in {"auto", "default"}:
        # The commit itself determines the harness implementation.
        # Fallback hierarchy matches repo evolution: H3 (Multi-Agent Specialist) -> H2 (Planner+Verifier) -> H1 (Hermes).
        try:
            return LangGraphH3Harness(cfg)
        except Exception:
            pass
        try:
            return LangGraphH2Harness(cfg)
        except Exception:
            pass
        return DirectHermesHarness(cfg)
    raise RuntimeError(f"Unknown harness id {harness_id!r}; expected h1, h2, h3, or auto")
