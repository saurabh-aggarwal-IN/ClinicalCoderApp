from __future__ import annotations

from clinical_coder.harnesses.base import BaseHarness
from clinical_coder.harnesses.direct_hermes import DirectHermesHarness
from clinical_coder.settings import Settings


def build_harness(cfg: Settings) -> BaseHarness:
    harness_id = cfg.harness_id.strip().lower()
    if harness_id == "h1":
        return DirectHermesHarness(cfg)
    if harness_id in {"h2", "h3"}:
        raise RuntimeError(
            f"Harness {harness_id!r} is not present at this commit. Check out the "
            "commit that adds it, or run H1."
        )
    raise RuntimeError(f"Unknown harness id {harness_id!r}; expected h1, h2, or h3")
