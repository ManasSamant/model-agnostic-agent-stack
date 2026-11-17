"""Model-agnostic LLM layer.

A single factory returns a LangChain chat model for any supported provider.
The rest of the application depends only on LangChain's ``BaseChatModel``
interface -- never on a specific vendor SDK -- so swapping the underlying
model (or comparing several) is a one-line change.

Supported providers: Anthropic (Claude), OpenAI, and local models via Ollama.
"""
from __future__ import annotations

import os
from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS = ("anthropic", "openai", "ollama")

# Sensible defaults per provider. Adjust to the exact model names available
# to you / your organisation.
DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


def get_llm(
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Return a LangChain chat model for ``provider``.

    Parameters
    ----------
    provider : one of ``SUPPORTED_PROVIDERS``.
    model    : optional explicit model name; falls back to ``DEFAULT_MODELS``.
    temperature : sampling temperature (0.0 = deterministic).
    """
    provider = provider.lower().strip()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        # Ollama runs locally (https://ollama.com) -- no API key required.
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def available_providers() -> list[str]:
    """Return providers that look usable from the current environment."""
    usable = []
    if os.getenv("ANTHROPIC_API_KEY"):
        usable.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        usable.append("openai")
    # Ollama is assumed available if installed locally.
    usable.append("ollama")
    return usable
