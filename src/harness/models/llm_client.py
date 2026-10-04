"""Thin wrapper around the Anthropic SDK.

Mirrors the real/mock split `get_decision_client` uses for Jev (see
`decision.py`): tests and local dry-runs use `LLMClient(mock=True)` so they
are deterministic, free, and don't need a network connection. Classifying
labels and extracting a confidence from the response text is Phase 2 work --
this wrapper just measures and returns one completion.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """One completion, with what's needed to compute latency and cost later."""

    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class LLMClient:
    def __init__(self, api_key: str, model: str, mock: bool = False) -> None:
        if not mock and not api_key:
            raise ValueError("api_key is required unless mock=True")
        self.model = model
        self._mock = mock
        self._client = None if mock else _build_anthropic_client(api_key)

    def complete(self, prompt: str, max_tokens: int = 256) -> LLMResponse:
        if self._mock:
            return _mock_complete(prompt)

        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def _build_anthropic_client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def _mock_complete(prompt: str) -> LLMResponse:
    """Deterministic stand-in: a hash of the prompt stands in for the model's text."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return LLMResponse(
        text=f"mock-response-{digest}",
        latency_seconds=0.0,
        input_tokens=len(prompt.split()),
        output_tokens=3,
    )
