"""LangGraph state object shared across all nodes in the workflow."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from app.models.schemas import Citation, ExecutionPlan, RetrievedItem


class GraphState(TypedDict, total=False):
    # Input
    user_query: str
    uploaded_documents: list[str]  # document_ids in scope for this query

    # Planning
    plan: Optional[ExecutionPlan]

    # Retrieval / analysis
    retrieved_context: dict[str, list[RetrievedItem]]  # {"text": [...], "figures": [...], "tables": [...]}
    agent_results: dict[str, str]  # e.g. {"researcher": "...", "multimodal": "..."}

    # Tools
    tool_results: dict[str, str]  # {"tool": name, "result": text}

    # Output
    final_answer: str
    citations: list[Citation]

    # Review / control flow
    review_status: str  # "pending" | "approved" | "rejected"
    review_reason: str
    retry_count: int

    # Observability
    agent_trace: list[str]
    tools_called: list[str]
