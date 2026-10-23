from __future__ import annotations

from clinical_coder.harnesses.base import BaseHarness
from clinical_coder.harnesses.direct_hermes import DirectHermesHarness
from clinical_coder.harnesses.langgraph_h2 import LangGraphH2Harness
from clinical_coder.settings import Settings


def build_harness(cfg: Settings) -> BaseHarness:
    harness_id = cfg.harness_id.strip().lower()
    if harness_id == "h1":
        return DirectHermesHarness(cfg)
    if harness_id == "h2":
        return LangGraphH2Harness(cfg)
    if harness_id == "h3":
        raise RuntimeError(
            "Harness h3 is not present at this commit. Check out the commit that "
            "adds the verifier, or run H1/H2."
        )
    raise RuntimeError(f"Unknown harness id {harness_id!r}; expected h1, h2, or h3")
