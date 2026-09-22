"""
Multimodal Analysis Agent.

Combines text context with figure and table context to answer questions
that reference visual content (figures, diagrams, charts, tables).
"""

from __future__ import annotations

from app.agents.llm import chat
from app.logging_config import get_logger
from app.models.schemas import RetrievedItem

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the Multimodal Analysis Agent. You answer questions that involve
figures, diagrams, charts, or tables from research papers, by combining:
1. Textual context retrieved from the papers
2. Generated descriptions of relevant figures
3. Structured (markdown) content of relevant tables

Rules:
- Ground your answer only in the provided context blocks.
- When describing a figure, rely on its generated description and caption — do not
  invent visual details that are not stated.
- When describing a table, read values directly from the provided markdown table.
- Clearly say when a referenced figure/table was not found in the retrieved context.
- Always mention which paper, page number, and figure/table number your answer is based on.
"""


def _format_items(items: list[RetrievedItem], label: str) -> str:
    if not items:
        return f"(no {label} context retrieved)"
    blocks = []
    for i, item in enumerate(items, start=1):
        extra = item.extra
        ref = extra.get("figure_number") or extra.get("table_number") or ""
        blocks.append(
            f"[{label.title()} {i}] file={item.filename} page={item.page_number} "
            f"number={ref}\n{item.text}"
        )
    return "\n\n".join(blocks)


def analyze_multimodal(
    user_query: str,
    text_items: list[RetrievedItem],
    figure_items: list[RetrievedItem],
    table_items: list[RetrievedItem],
) -> str:
    """Answer a question that involves figures and/or tables, combined with text context."""
    combined_context = "\n\n".join(
        [
            "== TEXT CONTEXT ==\n" + _format_items(text_items, "text"),
            "== FIGURE CONTEXT ==\n" + _format_items(figure_items, "figure"),
            "== TABLE CONTEXT ==\n" + _format_items(table_items, "table"),
        ]
    )

    user_prompt = (
        f"User question:\n{user_query}\n\n{combined_context}\n\n"
        "Write the answer now, grounded strictly in the context above."
    )
    answer = chat(_SYSTEM_PROMPT, user_prompt, temperature=0.2)
    logger.info("Multimodal Agent produced answer (%d chars)", len(answer))
    return answer
