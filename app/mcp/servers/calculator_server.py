"""
MCP server exposing the `calculate` tool.

Run standalone with:
    python -m app.mcp.servers.calculator_server
Communicates over stdio, per the standard MCP transport for locally-spawned
tool servers.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.tools.calculator import CalculatorError, calculate

mcp = FastMCP("calculator-server")


@mcp.tool()
def calculate_expression(expression: str) -> str:
    """
    Evaluate a numeric expression safely, e.g. '(0.31 + 0.28 + 0.35) / 3'
    or 'average(0.31, 0.28, 0.35)'. Returns the numeric result as a string.
    """
    try:
        result = calculate(expression)
        return str(result)
    except CalculatorError as exc:
        return f"ERROR: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
