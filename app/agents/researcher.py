"""
Research Analyst Agent.

Analyzes retrieved text context: summarizes, extracts methodology/dataset/
architecture/results/limitations, and compares across papers when multiple
documents are involved. Always grounds its answer in the provided context
and states clearly when information is not found.
"""

from __future__ import annotations

from app.agents.llm import chat
from app.logging_config import get_logger
from app.models.schemas import RetrievedItem

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the Research Analyst Agent in a multi-agent research-paper
assistant. You answer using ONLY the provided context chunks, each tagged with its
source (filename, page number, section).

Rules:
- Ground every claim in the given context. Do not use outside knowledge about the
  papers beyond what is provided.
- If the context does not contain the information needed to answer, say so clearly
  ("This was not found in the uploaded documents.") instead of guessing.
- When asked to compare multiple papers, organize the answer by dimension
  (e.g. methodology, dataset, results) and name which paper each point comes from.
- Keep the answer focused and well-structured. Use short paragraphs or bullet points.
- Do NOT fabricate page numbers, filenames, or citations — only refer to sources that
  literally appear in the provided context.
"""


def _format_context(items: list[RetrievedItem]) -> str:
    if not items:
        return "(no text context retrieved)"
    blocks = []
    for i, item in enumerate(items, start=1):
        blocks.append(
            f"[Source {i}] file={item.filename} page={item.page_number} "
            f"section={item.section or 'Unknown'}\n{item.text}"
        )
    return "\n\n".join(blocks)


def analyze(user_query: str, context_items: list[RetrievedItem], extra_notes: str = "") -> str:
    """Produce a grounded analytical answer to the user's query."""
    context_block = _format_context(context_items)
    user_prompt = (
        f"User question:\n{user_query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        + (f"Additional notes from other agents:\n{extra_notes}\n\n" if extra_notes else "")
        + "Write the answer now, grounded strictly in the context above."
    )
    answer = chat(_SYSTEM_PROMPT, user_prompt, temperature=0.2)
    logger.info("Research Analyst produced answer (%d chars)", len(answer))
    return answer
