from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from clinical_coder.harnesses.langgraph_common import (
    GraphState,
    LangGraphHarness,
    coder_node,
    planner_node,
    verifier_node,
    verifier_route,
)


class LangGraphH2Harness(LangGraphHarness):
    harness_id = "h2"
    harness_name = "LangGraph Planner -> Coder -> Verifier"

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("planner", planner_node)
        workflow.add_node("coder", coder_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "verifier")
        workflow.add_conditional_edges(
            "verifier",
            verifier_route,
            {"coder": "coder", "__end__": END},
        )
        return workflow.compile()
