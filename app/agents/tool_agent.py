"""
Tool Agent.

Decides whether an external tool is required, selects the appropriate one,
executes it through the MCP client (Tool Agent -> MCP Client -> MCP Server
-> Tool -> Result), and returns the result to the workflow. Tools are never
called unnecessarily — only when the plan indicates one is required.
"""

from __future__ import annotations

from app.agents.llm import chat
from app.logging_config import get_logger
from app.mcp import client as mcp_client
from app.models.schemas import ExecutionPlan, RetrievedItem

logger = get_logger(__name__)

_EXPRESSION_PROMPT = """Extract a single arithmetic expression that answers the user's
calculation request, using only numeric values found in the provided context.
Respond with ONLY the expression (e.g. "(0.31 + 0.28 + 0.35) / 3" or
"average(0.31, 0.28, 0.35)"), no explanation. If no relevant numbers are found in the
context, respond with exactly: NONE
"""


def extract_expression(user_query: str, context_items: list[RetrievedItem]) -> str:
    """Ask the LLM to turn a natural-language calculation request into an expression."""
    context_text = "\n".join(f"- {item.text[:300]}" for item in context_items) or "(no context)"
    user_prompt = f"User request: {user_query}\n\nContext:\n{context_text}"
    expression = chat(_EXPRESSION_PROMPT, user_prompt, temperature=0.0).strip()
    return "" if expression.upper() == "NONE" else expression


async def run_tool(plan: ExecutionPlan, user_query: str, numeric_hint: str = "") -> dict[str, str]:
    """
    Execute the tool indicated by the plan and return {"tool": name, "result": text}.
    Returns an empty dict if no tool is required.
    """
    if not plan.requires_tool or not plan.tool_hint:
        return {}

    tool_name = plan.tool_hint
    logger.info("Tool Agent invoking tool=%s", tool_name)

    if tool_name == "calculator":
        expression = numeric_hint or user_query
        result = await mcp_client.calculate(expression)
        return {"tool": "calculator", "result": result}

    if tool_name == "search_web":
        result = await mcp_client.search_web(user_query)
        return {"tool": "search_web", "result": result}

    logger.warning("Unknown tool_hint '%s'; skipping tool execution.", tool_name)
    return {}
