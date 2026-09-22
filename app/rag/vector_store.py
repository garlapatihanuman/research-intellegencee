"""
Vector store abstraction.

`VectorStore` defines the interface the rest of the app depends on.
`ChromaVectorStore` is the concrete implementation used today. A
`QdrantVectorStore` can be added later behind the same interface without
touching any calling code — swap it in `get_vector_store()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import ContentType, FigureRecord, TableRecord, TextChunk

logger = get_logger(__name__)


class VectorStore(ABC):
    """Interface every vector store backend must implement."""

    @abstractmethod
    def add_text_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def add_figures(self, figures: list[FigureRecord], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def add_tables(self, tables: list[TableRecord], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        content_types: Optional[list[ContentType]] = None,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed implementation, storing text/figure/table content in one collection."""

    def __init__(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- writes ---------------------------------------------------------- #

    def add_text_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "section": c.section,
                    "content_type": c.content_type.value,
                    "source": c.source,
                }
                for c in chunks
            ],
        )

    def add_figures(self, figures: list[FigureRecord], embeddings: list[list[float]]) -> None:
        if not figures:
            return
        self._collection.add(
            ids=[f.figure_id for f in figures],
            embeddings=embeddings,
            documents=[f.generated_description or f.caption for f in figures],
            metadatas=[
                {
                    "document_id": f.document_id,
                    "filename": f.filename,
                    "page_number": f.page_number,
                    "figure_number": f.figure_number,
                    "caption": f.caption,
                    "image_path": f.image_path,
                    "content_type": ContentType.FIGURE.value,
                }
                for f in figures
            ],
        )

    def add_tables(self, tables: list[TableRecord], embeddings: list[list[float]]) -> None:
        if not tables:
            return
        self._collection.add(
            ids=[t.table_id for t in tables],
            embeddings=embeddings,
            documents=[t.structured_content or t.caption for t in tables],
            metadatas=[
                {
                    "document_id": t.document_id,
                    "filename": t.filename,
                    "page_number": t.page_number,
                    "table_number": t.table_number,
                    "caption": t.caption,
                    "content_type": ContentType.TABLE.value,
                }
                for t in tables
            ],
        )

    # -- reads ------------------------------------------------------------ #

    def query(
        self,
        embedding: list[float],
        top_k: int,
        content_types: Optional[list[ContentType]] = None,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        filters = []
        if content_types:
            filters.append({"content_type": {"$in": [c.value for c in content_types]}})
        if document_ids:
            filters.append({"document_id": {"$in": document_ids}})
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        if not results.get("ids") or not results["ids"][0]:
            return output

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            output.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - distance,  # cosine distance -> similarity
                }
            )
        return output

    def clear(self) -> None:
        logger.warning("Clearing vector store collection: %s", settings.chroma_collection_name)
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_documents(self) -> list[dict[str, Any]]:
        all_items = self._collection.get(include=["metadatas"])
        seen: dict[str, dict[str, Any]] = {}
        for meta in all_items.get("metadatas", []):
            doc_id = meta.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {"document_id": doc_id, "filename": meta.get("filename")}
        return list(seen.values())

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})


_VECTOR_STORE_SINGLETON: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Factory selecting the configured vector store backend."""
    global _VECTOR_STORE_SINGLETON
    if _VECTOR_STORE_SINGLETON is not None:
        return _VECTOR_STORE_SINGLETON

    if settings.vector_store_provider == "chroma":
        _VECTOR_STORE_SINGLETON = ChromaVectorStore()
    elif settings.vector_store_provider == "qdrant":
        raise NotImplementedError(
            "Qdrant backend not yet implemented. Implement QdrantVectorStore(VectorStore) "
            "in this module and wire it up here — the rest of the app already depends "
            "only on the VectorStore interface."
        )
    else:
        raise ValueError(f"Unknown VECTOR_STORE_PROVIDER: {settings.vector_store_provider}")

    return _VECTOR_STORE_SINGLETON
