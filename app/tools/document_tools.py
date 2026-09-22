"""
Document-level tools: semantic search over ingested papers, and raw text
retrieval for a specific (document, page) pair — used when an agent needs
to double-check the exact page a claim is coming from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from app.config import settings
from app.models.schemas import RetrievedItem
from app.rag.retriever import retrieve


def search_documents(query: str, top_k: int = 6, document_ids: Optional[list[str]] = None) -> list[RetrievedItem]:
    """Semantic search across ingested documents (text content only)."""
    return retrieve(query, top_k=top_k, document_ids=document_ids)


def get_document_page(document_id: str, page_number: int) -> str:
    """Return the raw extracted text of a specific page of a stored document."""
    pdf_path = Path(settings.papers_dir) / f"{document_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"No stored PDF found for document_id={document_id}")

    doc = fitz.open(str(pdf_path))
    try:
        if not (1 <= page_number <= doc.page_count):
            raise ValueError(f"page_number {page_number} out of range (1-{doc.page_count})")
        return doc.load_page(page_number - 1).get_text("text")
    finally:
        doc.close()
