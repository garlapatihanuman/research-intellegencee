"""
LangGraph node functions. Each node reads from and writes to the shared
GraphState, and appends a short, human-readable entry to `agent_trace` so
the UI can show high-level execution progress without exposing any
internal chain-of-thought.
"""

from __future__ import annotations

import asyncio

from app.agents import multimodal as multimodal_agent
from app.agents import planner as planner_agent
from app.agents import researcher as researcher_agent
from app.agents import reviewer as reviewer_agent
from app.agents import tool_agent
from app.agents.retriever import run_retrieval
from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import Citation, ContentType, ExecutionPlan

logger = get_logger(__name__)


def _trace(state: dict, message: str) -> None:
    state.setdefault("agent_trace", []).append(message)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def planner_node(state: dict) -> dict:
    plan = planner_agent.plan_query(state["user_query"], state.get("uploaded_documents", []))
    state["plan"] = plan
    state.setdefault("retry_count", 0)
    _trace(state, f"Planner ✓ — {plan.query_type} query, {len(plan.tasks)} task(s) planned")
    return state


def retriever_node(state: dict) -> dict:
    plan: ExecutionPlan = state["plan"]
    context = run_retrieval(state["user_query"], plan, state.get("uploaded_documents"))
    state["retrieved_context"] = context
    n_text = len(context.get("text", []))
    n_fig = len(context.get("figures", []))
    n_tbl = len(context.get("tables", []))
    _trace(state, f"Retriever ✓ — {n_text} text, {n_fig} figure, {n_tbl} table chunk(s) retrieved")
    return state


def research_analyst_node(state: dict) -> dict:
    context = state.get("retrieved_context", {})
    text_items = context.get("text", [])

    tool_note = ""
    tool_results = state.get("tool_results") or {}
    if tool_results.get("result"):
        tool_note = f"Tool '{tool_results['tool']}' returned: {tool_results['result']}"

    answer = researcher_agent.analyze(state["user_query"], text_items, extra_notes=tool_note)
    state.setdefault("agent_results", {})["researcher"] = answer
    state["final_answer"] = answer
    _trace(state, "Research Analyst ✓ — draft answer produced")
    return state


def multimodal_node(state: dict) -> dict:
    plan: ExecutionPlan = state["plan"]
    if not plan.requires_multimodal:
        return state

    context = state.get("retrieved_context", {})
    answer = multimodal_agent.analyze_multimodal(
        state["user_query"],
        context.get("text", []),
        context.get("figures", []),
        context.get("tables", []),
    )
    state.setdefault("agent_results", {})["multimodal"] = answer
    state["final_answer"] = answer  # multimodal answer supersedes text-only draft
    _trace(state, "Multimodal Agent ✓ — visual content incorporated")
    return state


def tool_node(state: dict) -> dict:
    plan: ExecutionPlan = state["plan"]
    if not plan.requires_tool:
        return state

    numeric_hint = ""
    if plan.tool_hint == "calculator":
        context_items = state.get("retrieved_context", {}).get("text", [])
        numeric_hint = tool_agent.extract_expression(state["user_query"], context_items)
        if not numeric_hint:
            _trace(state, "Tool Agent ✓ — no numeric data found, calculator skipped")
            return state

    result = asyncio.run(tool_agent.run_tool(plan, state["user_query"], numeric_hint))
    if result:
        state["tool_results"] = result
        state.setdefault("tools_called", []).append(result["tool"])
        _trace(state, f"Tool Agent ✓ — called '{result['tool']}'")
    return state


def reviewer_node(state: dict) -> dict:
    context_items = state.get("retrieved_context", {}).get("text", [])
    context_items += state.get("retrieved_context", {}).get("figures", [])
    context_items += state.get("retrieved_context", {}).get("tables", [])

    result = reviewer_agent.review(state["user_query"], state.get("final_answer", ""), context_items)

    if result.approved or state.get("retry_count", 0) >= settings.max_review_retries:
        state["review_status"] = "approved"
        state["citations"] = _build_citations(state)
        _trace(state, "Reviewer ✓ — approved" + ("" if result.approved else " (max retries reached)"))
    else:
        state["review_status"] = "rejected"
        state["review_reason"] = result.reason
        state["retry_count"] = state.get("retry_count", 0) + 1
        _trace(state, f"Reviewer ✗ — rejected ({result.reason}); retrying retrieval")

    return state


def _build_citations(state: dict) -> list[Citation]:
    context = state.get("retrieved_context", {})
    citations: list[Citation] = []
    seen = set()
    for key in ("text", "figures", "tables"):
        for item in context.get(key, []):
            sig = (item.filename, item.page_number, item.content_type)
            if sig in seen:
                continue
            seen.add(sig)
            citations.append(
                Citation(
                    filename=item.filename,
                    page_number=item.page_number,
                    section=item.section,
                    content_type=ContentType(item.content_type),
                )
            )
    return citations


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #


def route_after_planner(state: dict) -> str:
    return "retriever"


def route_after_multimodal(state: dict) -> str:
    plan: ExecutionPlan = state["plan"]
    return "tool_agent" if plan.requires_tool else "reviewer"


def route_after_review(state: dict) -> str:
    return "end" if state.get("review_status") == "approved" else "retriever"
