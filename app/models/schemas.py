"""
Pydantic data models shared across the ingestion pipeline, agents,
LangGraph state, and API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"


# --------------------------------------------------------------------------- #
# Ingestion / chunk-level models
# --------------------------------------------------------------------------- #


class TextChunk(BaseModel):
    """A single chunk of extracted text with full provenance metadata."""

    document_id: str
    filename: str
    page_number: int
    section: str = "Unknown"
    chunk_id: str
    content_type: ContentType = ContentType.TEXT
    source: str = "pdf_text"
    text: str


class FigureRecord(BaseModel):
    """A figure/image extracted from a PDF page."""

    document_id: str
    filename: str
    page_number: int
    figure_number: int
    figure_id: str
    caption: str = ""
    image_path: str
    generated_description: str = ""
    content_type: ContentType = ContentType.FIGURE


class TableRecord(BaseModel):
    """A table extracted from a PDF page."""

    document_id: str
    filename: str
    page_number: int
    table_number: int
    table_id: str
    caption: str = ""
    structured_content: str = ""  # markdown/csv-like rendering of the table
    content_type: ContentType = ContentType.TABLE


class DocumentMetadata(BaseModel):
    """Top-level metadata for an ingested document."""

    document_id: str
    filename: str
    num_pages: int
    num_chunks: int = 0
    num_figures: int = 0
    num_tables: int = 0
    status: str = "processed"


class IngestionResult(BaseModel):
    document: DocumentMetadata
    chunks: list[TextChunk] = Field(default_factory=list)
    figures: list[FigureRecord] = Field(default_factory=list)
    tables: list[TableRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Retrieval models
# --------------------------------------------------------------------------- #


class RetrievedItem(BaseModel):
    """A single retrieved unit of context, regardless of modality."""

    content_type: ContentType
    text: str  # for figures/tables this is the description/structured content
    document_id: str
    filename: str
    page_number: int
    section: Optional[str] = None
    score: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    filename: str
    page_number: int
    section: Optional[str] = None
    content_type: ContentType = ContentType.TEXT


# --------------------------------------------------------------------------- #
# Planning models
# --------------------------------------------------------------------------- #


class ExecutionPlan(BaseModel):
    query_type: str = "simple"  # "simple" | "complex"
    tasks: list[str] = Field(default_factory=list)
    requires_multimodal: bool = False
    requires_tool: bool = False
    tool_hint: Optional[str] = None
    target_documents: list[str] = Field(default_factory=list)  # empty = all


# --------------------------------------------------------------------------- #
# Review models
# --------------------------------------------------------------------------- #


class ReviewResult(BaseModel):
    approved: bool
    reason: str = ""
    missing_information: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# API request / response models
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    query: str
    document_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    review_status: str
    agent_trace: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    retrieved_pages: list[int] = Field(default_factory=list)
