"""
Planner Agent.

Understands the user query, classifies it as simple/complex, decides which
downstream capabilities are needed (multimodal analysis, tools), and
produces a structured ExecutionPlan that the rest of the graph follows.
"""

from __future__ import annotations

from app.agents.llm import chat_json
from app.logging_config import get_logger
from app.models.schemas import ExecutionPlan

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the Planner Agent in a research-paper analysis system.

Given a user question about one or more uploaded research papers, produce a JSON
execution plan with this exact schema:

{
  "query_type": "simple" | "complex",
  "tasks": ["short imperative task description", ...],
  "requires_multimodal": true | false,
  "requires_tool": true | false,
  "tool_hint": "calculator" | "search_web" | null
}

Guidelines:
- "simple" = a single-paper factual question answerable from text retrieval alone.
- "complex" = comparisons across papers, multi-part questions, or anything needing
  several retrieval/analysis steps.
- requires_multimodal = true if the question references a figure, diagram, chart,
  image, or table.
- requires_tool = true only if the question needs a calculation (set tool_hint to
  "calculator") or explicitly asks to search for outside/additional information
  (set tool_hint to "search_web"). Otherwise false and tool_hint null.
- Keep "tasks" concrete and ordered, e.g. ["Retrieve methodology sections",
  "Retrieve dataset descriptions", "Compare across papers", "Generate final answer"].
- Respond with ONLY the JSON object, no commentary.
"""


def plan_query(user_query: str, known_document_ids: list[str]) -> ExecutionPlan:
    """Produce an ExecutionPlan for the given user query."""
    user_prompt = (
        f"User question: {user_query}\n\n"
        f"Known document IDs currently in the system: {known_document_ids or 'none uploaded yet'}\n"
        "Produce the JSON execution plan now."
    )

    data = chat_json(_SYSTEM_PROMPT, user_prompt)

    if not data:
        # Safe fallback: treat as a simple text-retrieval question.
        return ExecutionPlan(
            query_type="simple",
            tasks=["Retrieve relevant context", "Generate final response"],
            requires_multimodal=False,
            requires_tool=False,
        )

    plan = ExecutionPlan(
        query_type=data.get("query_type", "simple"),
        tasks=data.get("tasks") or ["Retrieve relevant context", "Generate final response"],
        requires_multimodal=bool(data.get("requires_multimodal", False)),
        requires_tool=bool(data.get("requires_tool", False)),
        tool_hint=data.get("tool_hint"),
    )
    logger.info("Plan produced: %s", plan.model_dump())
    return plan
