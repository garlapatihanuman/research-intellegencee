"""
Safe arithmetic calculator tool.

Uses Python's `ast` module to evaluate a restricted expression grammar
(numbers and + - * / ** % ( ) only) rather than `eval`, so it cannot execute
arbitrary code.
"""

from __future__ import annotations

import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Safely evaluate a numeric expression, e.g. 'sum([0.31, 0.28, 0.35]) / 3'."""
    # Support simple aggregate helpers without exposing full builtins.
    expression = expression.strip()
    if expression.lower().startswith(("average(", "avg(", "mean(")):
        inner = expression[expression.index("(") + 1 : expression.rindex(")")]
        numbers = [float(n.strip()) for n in inner.split(",") if n.strip()]
        if not numbers:
            raise CalculatorError("No numbers provided to average().")
        return sum(numbers) / len(numbers)

    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except CalculatorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CalculatorError(f"Could not evaluate expression '{expression}': {exc}") from exc
