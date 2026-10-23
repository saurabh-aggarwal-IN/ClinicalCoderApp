from __future__ import annotations

from clinical_coder.harnesses.langgraph_common import (
    LangGraphHarness,
    coder_node,
    compile_graph,
    planner_node,
)


class LangGraphH2Harness(LangGraphHarness):
    harness_id = "h2"
    harness_name = "LangGraph Planner -> Coder"

    def build_graph(self):
        return compile_graph(
            [("planner", planner_node), ("coder", coder_node)],
            [("planner", "coder"), ("coder", "__end__")],
        )
