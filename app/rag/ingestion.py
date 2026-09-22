"""
Orchestrates the full ingestion pipeline for a single PDF:

PDF -> validate -> parse -> chunk -> extract figures -> extract tables
    -> describe figures -> embed everything -> write to vector store
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import DocumentMetadata, IngestionResult
from app.multimodal.figure_analyzer import describe_figures
from app.multimodal.image_extractor import extract_figures
from app.multimodal.table_extractor import extract_tables
from app.rag.chunking import chunk_document
from app.rag.embeddings import embedding_service
from app.rag.parser import parse_pdf, validate_pdf
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 64


def _batched(items: list, batch_size: int) -> list[list]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def ingest_pdf(uploaded_file_path: str, original_filename: str) -> IngestionResult:
    """
    Run the complete ingestion pipeline for one uploaded PDF and persist all
    resulting chunks/figures/tables (with embeddings) into the vector store.
    """
    validate_pdf(uploaded_file_path)

    document_id = f"doc_{uuid.uuid4().hex[:10]}"

    # Persist a stable copy of the source PDF for later reference (e.g. get_document_page).
    stored_path = Path(settings.papers_dir) / f"{document_id}.pdf"
    stored_path.write_bytes(Path(uploaded_file_path).read_bytes())

    parsed = parse_pdf(str(stored_path), original_filename)
    chunks = chunk_document(document_id, parsed)
    figures = extract_figures(str(stored_path), document_id, original_filename)
    tables = extract_tables(str(stored_path), document_id, original_filename)

    figures = describe_figures(figures)

    store = get_vector_store()

    for batch in _batched(chunks, _EMBED_BATCH_SIZE):
        embeddings = embedding_service.embed_texts([c.text for c in batch])
        store.add_text_chunks(batch, embeddings)

    if figures:
        fig_texts = [f.generated_description or f.caption or "figure" for f in figures]
        for start in range(0, len(figures), _EMBED_BATCH_SIZE):
            batch = figures[start : start + _EMBED_BATCH_SIZE]
            embeddings = embedding_service.embed_texts(fig_texts[start : start + _EMBED_BATCH_SIZE])
            store.add_figures(batch, embeddings)

    if tables:
        table_texts = [t.structured_content or t.caption or "table" for t in tables]
        for start in range(0, len(tables), _EMBED_BATCH_SIZE):
            batch = tables[start : start + _EMBED_BATCH_SIZE]
            embeddings = embedding_service.embed_texts(table_texts[start : start + _EMBED_BATCH_SIZE])
            store.add_tables(batch, embeddings)

    metadata = DocumentMetadata(
        document_id=document_id,
        filename=original_filename,
        num_pages=parsed.num_pages,
        num_chunks=len(chunks),
        num_figures=len(figures),
        num_tables=len(tables),
        status="processed",
    )

    logger.info(
        "Ingestion complete for %s: %d chunks, %d figures, %d tables",
        original_filename,
        len(chunks),
        len(figures),
        len(tables),
    )

    return IngestionResult(document=metadata, chunks=chunks, figures=figures, tables=tables)
