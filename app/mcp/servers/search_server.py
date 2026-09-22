"""
MCP server exposing the `search_web` tool. Run standalone with:
    python -m app.mcp.servers.search_server
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from app.tools.search import search_web

mcp = FastMCP("search-server")


@mcp.tool()
def search_web_tool(query: str, num_results: int = 5) -> str:
    """Search the web for information not found in the uploaded papers. Returns JSON results."""
    results = search_web(query, num_results=num_results)
    return json.dumps(results, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
