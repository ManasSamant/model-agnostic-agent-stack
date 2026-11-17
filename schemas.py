"""Structured-output schemas.

These Pydantic models are passed to ``llm.with_structured_output(...)`` so the
model returns validated, typed JSON instead of free text -- one of the shared
building blocks every internal tool can rely on.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """A structured answer returned by the agent."""

    answer: str = Field(description="The final answer to the user's question.")
    confidence: float = Field(
        description="Model's confidence in the answer, between 0 and 1.",
        ge=0.0,
        le=1.0,
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Tools or sources the agent used to reach the answer.",
    )


class JudgeVerdict(BaseModel):
    """Structured verdict from the LLM-as-judge evaluation."""

    score: float = Field(
        description="Quality score of the answer, between 0 and 1.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="One short sentence explaining the score.")
