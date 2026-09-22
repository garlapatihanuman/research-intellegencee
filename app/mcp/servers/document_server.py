"""
MCP server exposing document-related tools: `search_documents` and
`get_document_page`. Run standalone with:
    python -m app.mcp.servers.document_server
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from app.tools.document_tools import get_document_page, search_documents

mcp = FastMCP("document-server")


@mcp.tool()
def search_documents_tool(query: str, top_k: int = 6) -> str:
    """
    Semantic search across all ingested research papers. Returns a JSON list
    of matches, each with filename, page_number, section, content_type,
    score, and a text snippet.
    """
    results = search_documents(query, top_k=top_k)
    payload = [
        {
            "filename": r.filename,
            "page_number": r.page_number,
            "section": r.section,
            "content_type": r.content_type.value,
            "score": round(r.score, 4),
            "text": r.text[:500],
        }
        for r in results
    ]
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_document_page_tool(document_id: str, page_number: int) -> str:
    """Return the raw extracted text of a specific page of a stored document."""
    try:
        return get_document_page(document_id, page_number)
    except (FileNotFoundError, ValueError) as exc:
        return f"ERROR: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
