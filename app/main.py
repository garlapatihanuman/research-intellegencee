"""
FastAPI application: exposes document upload/ingestion, document listing,
vector-store clearing, and the main query endpoint that runs the LangGraph
multi-agent workflow.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.graph.workflow import run_query
from app.logging_config import get_logger
from app.models.schemas import (
    Citation,
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
)
from app.rag.ingestion import ingest_pdf
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)

app = FastAPI(
    title="AI Research Paper Intelligence & Multimodal Analysis System",
    version="1.0.0",
    description="Upload research papers and ask questions using a RAG + multi-agent LangGraph system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentMetadata)
async def upload_document(file: UploadFile = File(...)) -> DocumentMetadata:
    """Upload and ingest a single PDF research paper."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path, file.filename)
        return result.document
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/documents")
def list_documents() -> list[dict]:
    """List all documents currently stored in the vector database."""
    return get_vector_store().list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    get_vector_store().delete_document(document_id)
    return {"deleted": document_id}


@app.post("/documents/clear")
def clear_documents() -> dict:
    """Clear the entire vector database."""
    get_vector_store().clear()
    return {"status": "cleared"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full multi-agent workflow for a user question."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        final_state = run_query(request.query, request.document_ids or None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow execution failed")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

    citations = final_state.get("citations", [])
    retrieved_pages = sorted(
        {
            item.page_number
            for items in final_state.get("retrieved_context", {}).values()
            for item in items
        }
    )

    return QueryResponse(
        answer=final_state.get("final_answer", ""),
        citations=[c if isinstance(c, Citation) else Citation(**c) for c in citations],
        review_status=final_state.get("review_status", "unknown"),
        agent_trace=final_state.get("agent_trace", []),
        tools_called=final_state.get("tools_called", []),
        retrieved_pages=retrieved_pages,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
