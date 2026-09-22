"""
LangGraph orchestration.

Builds the controlled agent graph:

START -> Planner -> Retriever -> Research Analyst -> Multimodal Agent
      -> [Tool Agent if required] -> Reviewer
      -> (approved) -> END
      -> (rejected, retries remain) -> Retriever

The review loop is bounded by settings.max_review_retries (enforced inside
reviewer_node) to prevent uncontrolled looping.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph import nodes
from app.graph.state import GraphState
from app.logging_config import get_logger

logger = get_logger(__name__)


def build_workflow():
    graph = StateGraph(GraphState)

    graph.add_node("planner", nodes.planner_node)
    graph.add_node("retriever", nodes.retriever_node)
    graph.add_node("research_analyst", nodes.research_analyst_node)
    graph.add_node("multimodal_agent", nodes.multimodal_node)
    graph.add_node("tool_agent", nodes.tool_node)
    graph.add_node("reviewer", nodes.reviewer_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "research_analyst")
    graph.add_edge("research_analyst", "multimodal_agent")

    graph.add_conditional_edges(
        "multimodal_agent",
        nodes.route_after_multimodal,
        {"tool_agent": "tool_agent", "reviewer": "reviewer"},
    )
    graph.add_edge("tool_agent", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        nodes.route_after_review,
        {"end": END, "retriever": "retriever"},
    )

    return graph.compile()


_WORKFLOW = None


def get_workflow():
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = build_workflow()
    return _WORKFLOW


def run_query(user_query: str, document_ids: list[str] | None = None) -> dict:
    """Run the full multi-agent workflow for a single user query."""
    workflow = get_workflow()
    initial_state: dict = {
        "user_query": user_query,
        "uploaded_documents": document_ids or [],
        "agent_trace": [],
        "tools_called": [],
        "retry_count": 0,
    }
    logger.info("Running workflow for query: %.80s", user_query)
    final_state = workflow.invoke(initial_state, config={"recursion_limit": 50})
    return final_state
