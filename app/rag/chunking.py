"""
Text chunking. Uses LangChain's RecursiveCharacterTextSplitter so chunk
boundaries respect paragraph/sentence structure where possible.
"""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models.schemas import ContentType, TextChunk
from app.rag.parser import ParsedDocument


def chunk_document(document_id: str, parsed: ParsedDocument) -> list[TextChunk]:
    """
    Split every page's text into overlapping chunks while preserving
    page number and detected section as metadata on each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    for page in parsed.pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            if not piece.strip():
                continue
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    filename=parsed.filename,
                    page_number=page.page_number,
                    section=page.section,
                    chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                    content_type=ContentType.TEXT,
                    source="pdf_text",
                    text=piece.strip(),
                )
            )
    return chunks
