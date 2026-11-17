"""CLI entry point.

Examples:
    python main.py "What is 15 * (8 + 2)?"
    python main.py "Tell me about MCP" --provider ollama
    python main.py "Summarise LangChain in one line" --structured
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.agent import answer_structured, run_agent

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model-agnostic agent.")
    parser.add_argument("query", help="The question to ask the agent.")
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="LLM provider: anthropic | openai | ollama (default: anthropic).",
    )
    parser.add_argument("--model", default=None, help="Override the model name.")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Return a validated structured Answer instead of free text.",
    )
    args = parser.parse_args()

    if args.structured:
        result = answer_structured(args.query, provider=args.provider, model=args.model)
        print(result.model_dump_json(indent=2))
    else:
        print(run_agent(args.query, provider=args.provider, model=args.model))


if __name__ == "__main__":
    main()
