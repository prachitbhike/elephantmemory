"""Thin wrappers around Anthropic + OpenAI with built-in cost tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
import openai

from .cost import cost_usd

ASSISTANT_MODEL = os.getenv("ELEPHANT_ASSISTANT_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("ELEPHANT_JUDGE_MODEL", "claude-sonnet-4-6")
EMBED_MODEL = os.getenv("ELEPHANT_EMBED_MODEL", "text-embedding-3-small")


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str


_anthropic_client: anthropic.Anthropic | None = None
_openai_client: openai.OpenAI | None = None


def anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


def chat(
    system: str,
    messages: list[dict],
    *,
    model: str = ASSISTANT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> LLMResponse:
    resp = anthropic_client().messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    tin = resp.usage.input_tokens
    tout = resp.usage.output_tokens
    return LLMResponse(text, tin, tout, cost_usd(model, tin, tout), model)


def embed(texts: list[str], *, model: str = EMBED_MODEL) -> tuple[list[list[float]], float]:
    resp = openai_client().embeddings.create(model=model, input=texts)
    total_in = resp.usage.prompt_tokens
    return [d.embedding for d in resp.data], cost_usd(model, total_in, 0)
