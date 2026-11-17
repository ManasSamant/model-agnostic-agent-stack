# Model-Agnostic Agent Stack

A small but production-shaped **LangChain + LangGraph** reference stack: one
**model-agnostic LLM layer**, shared building blocks (**tool calling**, **MCP**,
**structured outputs**), and an **eval + tracing** harness to compare LLM
providers and decide which model fits which use case.

The idea: internal tools shouldn't each wire up their own LLM differently. They
should build on one shared foundation where the model is swappable and every run
is traced and evaluable. This repo is a minimal version of exactly that.

## What's inside

```
model-agnostic-agent-stack/
├── main.py                 # CLI: run the agent (free-text or structured output)
├── requirements.txt
├── .env.example            # copy to .env and fill in keys
├── examples/
│   └── eval_dataset.json   # tiny eval set (question + reference answer)
└── src/
    ├── llm_layer.py        # model-agnostic factory: get_llm("anthropic" | "openai" | "ollama")
    ├── tools.py            # @tool functions the agent can call
    ├── schemas.py          # Pydantic schemas for structured outputs
    ├── agent.py            # LangGraph ReAct agent wiring model + tools
    ├── evals.py            # LLM-as-judge: compare providers, with LangSmith tracing
    └── mcp_tools.py        # optional: load tools from an MCP server
```

## Core ideas

- **Model-agnostic layer** (`src/llm_layer.py`) — one `get_llm(provider)` returns
  a LangChain `BaseChatModel`, so the whole stack runs on **Claude, GPT, or a
  local Ollama model** by changing a single argument. Nothing downstream depends
  on a vendor SDK.
- **LangGraph agent** (`src/agent.py`) — `create_react_agent` runs the
  reason/act loop over the tools as a graph.
- **Tool calling** (`src/tools.py`) — plain Python functions decorated with
  `@tool` (a safe calculator and a mock internal knowledge-base lookup). Swap in
  your own tools (CRM lookups, DB queries, ...).
- **Structured outputs** (`src/schemas.py`) — `llm.with_structured_output(Answer)`
  returns validated, typed JSON instead of free text.
- **MCP** (`src/mcp_tools.py`, optional) — load tools from any MCP server via
  `langchain-mcp-adapters` and hand them to the same agent.
- **Evals + tracing** (`src/evals.py`) — an **LLM-as-judge** scores each
  provider's answers against references, so you can compare models objectively.
  Set the LangSmith env vars and every model/tool call is traced automatically.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# edit .env and add at least one provider key (e.g. ANTHROPIC_API_KEY)
```

For the **Ollama** provider, install Ollama from <https://ollama.com> and pull a
model, e.g. `ollama pull llama3.1`. No API key is needed for local models.

> **Model names** in `src/llm_layer.py` (`DEFAULT_MODELS`) are sensible defaults —
> adjust them to the exact models available to you.

## Usage

```bash
# Ask the agent (uses tools when helpful)
python main.py "What is 15 * (8 + 2)?"

# Run on a different provider
python main.py "Tell me about MCP" --provider ollama

# Get a validated, structured answer
python main.py "Summarise LangChain in one line" --structured
```

Compare providers on the eval set:

```bash
python -m src.evals --providers "anthropic,ollama"
```

Example output:

```
[anthropic] 'What is 15 * (8 + 2)?' -> score 1.00 (Correct, used the calculator tool)
[ollama]    'What is 15 * (8 + 2)?' -> score 1.00 (Correct)
...
=== Average scores ===
anthropic    0.960
ollama       0.780
```

## Tracing with LangSmith (optional)

Set these in `.env` and runs are traced automatically — no code changes:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=model-agnostic-agent-stack
```

Open the project in the LangSmith UI to inspect each model call, tool call and
token count.

## MCP (optional)

`src/mcp_tools.py` shows how to load tools from an MCP server (example: the
filesystem server) and give them to the same LangGraph agent:

```bash
python -m src.mcp_tools
```

## Possible next steps

- Add a real tool (HubSpot / Postgres / internal API).
- Add more eval cases and a second judge for agreement.
- Add a LangGraph node that enforces `structured_output` after tool use.

---

Built as a reference for consolidating many ad-hoc LLM integrations onto one
shared LangChain/LangGraph foundation.
