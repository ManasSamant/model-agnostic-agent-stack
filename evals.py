"""Compare LLM providers with an LLM-as-judge evaluation.

For each question in the dataset, every provider's agent produces an answer,
and a judge model scores it against the reference. This is how you decide
"which model fits which use case".

LangSmith tracing turns on automatically when ``LANGCHAIN_TRACING_V2=true`` and
``LANGCHAIN_API_KEY`` are set -- every model and tool call is then traced in the
LangSmith UI with no extra code.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import run_agent
from .llm_layer import get_llm
from .schemas import JudgeVerdict

_JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a question, a candidate answer and a "
    "reference answer, score how correct and useful the candidate answer is "
    "from 0 (wrong) to 1 (fully correct). Return a score and one short reason."
)


def judge(question: str, answer: str, reference: str, provider: str = "anthropic") -> JudgeVerdict:
    """Score a single answer against a reference using an LLM-as-judge."""
    llm = get_llm(provider).with_structured_output(JudgeVerdict)
    prompt = (
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Reference answer: {reference}\n"
    )
    return llm.invoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)])


def run_evals(dataset_path: str, providers: list[str], judge_provider: str = "anthropic") -> dict:
    """Run every provider over the dataset and return average scores.

    Returns a dict like ``{"anthropic": 0.92, "ollama": 0.71}``.
    """
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    results: dict[str, list[float]] = {p: [] for p in providers}

    for item in dataset:
        question, reference = item["question"], item["reference"]
        for provider in providers:
            answer = run_agent(question, provider=provider)
            verdict = judge(question, answer, reference, provider=judge_provider)
            results[provider].append(verdict.score)
            print(f"[{provider}] {question!r} -> score {verdict.score:.2f} ({verdict.reasoning})")

    return {
        provider: (sum(scores) / len(scores) if scores else 0.0)
        for provider, scores in results.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LLM providers with evals.")
    parser.add_argument("--dataset", default="examples/eval_dataset.json")
    parser.add_argument(
        "--providers", default="anthropic", help="comma-separated, e.g. 'anthropic,ollama'"
    )
    args = parser.parse_args()

    scores = run_evals(args.dataset, [p.strip() for p in args.providers.split(",")])
    print("\n=== Average scores ===")
    for provider, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{provider:12s} {score:.3f}")
