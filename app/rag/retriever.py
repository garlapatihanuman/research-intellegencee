"""
Semantic retrieval: embeds a query and searches the vector store, optionally
filtered by content type and/or document id, returning normalized
RetrievedItem objects with full provenance for citation generation.
"""

from __future__ import annotations

from typing import Optional

from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import ContentType, RetrievedItem
from app.rag.embeddings import embedding_service
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    content_types: Optional[list[ContentType]] = None,
    document_ids: Optional[list[str]] = None,
) -> list[RetrievedItem]:
    """Embed `query` and return the top-k most relevant items from the store."""
    top_k = top_k or settings.top_k_retrieval
    query_embedding = embedding_service.embed_query(query)

    store = get_vector_store()
    raw_results = store.query(
        embedding=query_embedding,
        top_k=top_k,
        content_types=content_types,
        document_ids=document_ids,
    )

    items: list[RetrievedItem] = []
    for result in raw_results:
        meta = result["metadata"]
        items.append(
            RetrievedItem(
                content_type=ContentType(meta.get("content_type", "text")),
                text=result["text"],
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                page_number=int(meta.get("page_number", 0)),
                section=meta.get("section"),
                score=result.get("score", 0.0),
                extra={k: v for k, v in meta.items() if k not in {"document_id", "filename", "page_number", "section", "content_type"}},
            )
        )

    logger.info("Retrieved %d items for query: %.60s", len(items), query)
    return items


def retrieve_multimodal(
    query: str, top_k: Optional[int] = None, document_ids: Optional[list[str]] = None
) -> dict[str, list[RetrievedItem]]:
    """Retrieve text, figure, and table context separately for a multimodal question."""
    top_k = top_k or settings.top_k_retrieval
    return {
        "text": retrieve(query, top_k=top_k, content_types=[ContentType.TEXT], document_ids=document_ids),
        "figures": retrieve(query, top_k=max(2, top_k // 2), content_types=[ContentType.FIGURE], document_ids=document_ids),
        "tables": retrieve(query, top_k=max(2, top_k // 2), content_types=[ContentType.TABLE], document_ids=document_ids),
    }
