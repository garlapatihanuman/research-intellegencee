"""
Retriever Agent.

Wraps the RAG retrieval layer: converts the query into embeddings, searches
the vector store, and returns source-grounded text (and, when requested,
figure/table) context.
"""

from __future__ import annotations

from app.logging_config import get_logger
from app.models.schemas import ExecutionPlan, RetrievedItem
from app.rag.retriever import retrieve, retrieve_multimodal

logger = get_logger(__name__)


def run_retrieval(
    user_query: str, plan: ExecutionPlan, document_ids: list[str] | None = None
) -> dict[str, list[RetrievedItem]]:
    """
    Retrieve context for the query. Always retrieves text; also retrieves
    figures/tables when the plan indicates a multimodal question.
    """
    target_docs = document_ids or (plan.target_documents or None)

    if plan.requires_multimodal:
        results = retrieve_multimodal(user_query, document_ids=target_docs)
    else:
        results = {"text": retrieve(user_query, document_ids=target_docs), "figures": [], "tables": []}

    total = sum(len(v) for v in results.values())
    logger.info("Retriever Agent gathered %d total items", total)
    return results
