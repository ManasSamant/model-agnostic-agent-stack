"""Optional: load tools from an MCP server into the agent.

MCP (Model Context Protocol) is an open standard for exposing tools and data to
LLM apps. ``langchain-mcp-adapters`` converts MCP tools into LangChain tools so
a LangGraph agent can call them like any other tool.

This module is optional and async. It is not imported by ``main.py`` so the
core demo runs without any MCP server. To try it, install an MCP server (for
example the official filesystem server) and run ``python -m src.mcp_tools``.
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .llm_layer import get_llm


async def build_agent_with_mcp(provider: str = "anthropic"):
    """Build an agent whose tools come from one or more MCP servers."""
    # Imported lazily so the package installs/loads without MCP extras.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            # Example: the filesystem MCP server over stdio.
            # Requires: npx -y @modelcontextprotocol/server-filesystem <path>
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    llm = get_llm(provider)
    return create_react_agent(llm, tools)


async def _demo() -> None:
    agent = await build_agent_with_mcp()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="List the files in the current directory.")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(_demo())
