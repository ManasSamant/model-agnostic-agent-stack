"""The agent itself, built on LangGraph.

``create_react_agent`` wires the model-agnostic LLM to the tools and runs the
reason/act loop as a LangGraph graph. Because the model comes from
``llm_layer.get_llm``, the exact same agent runs on Claude, GPT or a local
Ollama model by changing one argument.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .llm_layer import get_llm
from .schemas import Answer
from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant for an internal tooling team. "
    "Use the available tools when they help, and be concise."
)


def build_agent(provider: str = "anthropic", model: str | None = None):
    """Build a LangGraph ReAct agent on the given provider."""
    llm = get_llm(provider, model)
    return create_react_agent(llm, TOOLS)


def run_agent(query: str, provider: str = "anthropic", model: str | None = None) -> str:
    """Run the agent on a query and return its final text answer."""
    agent = build_agent(provider, model)
    result = agent.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]}
    )
    return result["messages"][-1].content


def answer_structured(
    query: str, provider: str = "anthropic", model: str | None = None
) -> Answer:
    """Return a validated, structured ``Answer`` instead of free text.

    Demonstrates the shared 'structured outputs' building block. (This variant
    answers directly; combine with tools by passing the agent transcript in.)
    """
    llm = get_llm(provider, model).with_structured_output(Answer)
    return llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    )
