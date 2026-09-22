"""
MCP client wrapper.

The Tool Agent never imports app.tools.* directly — it goes through this
client, which spawns each MCP server as a stdio subprocess and calls tools
on it. This keeps the MCP layer cleanly separated from the LangGraph agents,
as required by the architecture: Tool Agent -> MCP Client -> MCP Server -> Tool.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.logging_config import get_logger

logger = get_logger(__name__)

_SERVER_MODULES = {
    "calculator": "app.mcp.servers.calculator_server",
    "document": "app.mcp.servers.document_server",
    "search": "app.mcp.servers.search_server",
}


@asynccontextmanager
async def _mcp_session(server_key: str):
    module = _SERVER_MODULES[server_key]
    params = StdioServerParameters(command=sys.executable, args=["-m", module])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(server_key: str, tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Spawn the given MCP server, call one tool on it, and return the text result.
    A fresh subprocess per call keeps this simple and robust for a portfolio
    project; for high-throughput production use, sessions would be pooled.
    """
    logger.info("MCP call -> server=%s tool=%s args=%s", server_key, tool_name, arguments)
    async with _mcp_session(server_key) as session:
        result = await session.call_tool(tool_name, arguments=arguments)
        text_parts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(text_parts)


async def calculate(expression: str) -> str:
    return await call_tool("calculator", "calculate_expression", {"expression": expression})


async def search_documents(query: str, top_k: int = 6) -> str:
    return await call_tool("document", "search_documents_tool", {"query": query, "top_k": top_k})


async def get_document_page(document_id: str, page_number: int) -> str:
    return await call_tool(
        "document", "get_document_page_tool", {"document_id": document_id, "page_number": page_number}
    )


async def search_web(query: str, num_results: int = 5) -> str:
    return await call_tool("search", "search_web_tool", {"query": query, "num_results": num_results})
