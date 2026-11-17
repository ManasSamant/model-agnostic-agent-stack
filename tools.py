"""Tools the agent can call.

Each tool is a plain Python function decorated with ``@tool``; LangChain /
LangGraph handle the tool-calling protocol with the model. Replace or extend
these with your own internal tools (HubSpot lookups, database queries, ...).
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

# --- safe arithmetic evaluator -------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.

    Supports + - * / ** % and parentheses. Use this for any math.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not evaluate '{expression}' ({exc})."


# --- a tiny mock "internal knowledge base" ------------------------------------

_COMPANY_FACTS = {
    "thorit": "Thorit is an AI & CRM consultancy and a Diamond HubSpot Partner.",
    "langchain": "LangChain is a framework for building applications with LLMs.",
    "mcp": "MCP (Model Context Protocol) is an open standard for connecting "
    "tools and data sources to LLM applications.",
}


@tool
def lookup_fact(topic: str) -> str:
    """Look up a short fact about a topic from the internal knowledge base.

    Use this when the user asks about an internal topic such as a tool or company.
    """
    return _COMPANY_FACTS.get(
        topic.lower().strip(), f"No internal record found for '{topic}'."
    )


# Export the toolset the agent is built with.
TOOLS = [calculator, lookup_fact]
