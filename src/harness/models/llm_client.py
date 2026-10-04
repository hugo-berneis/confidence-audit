"""Thin wrapper around the Anthropic SDK.

Mirrors the real/mock split `get_decision_client` uses for Jev (see
`decision.py`): tests and local dry-runs use `LLMClient(mock=True)` so they
are deterministic, free, and don't need a network connection.

`classify()` is this harness's one confidence-elicitation method for the
LLM (per CLAUDE.md's model rules: LLMs don't return calibrated probabilities
by default, so verbalized confidence -- asking the model to self-report a
0-100 confidence alongside its label -- is the one method implemented and
documented; it should be listed as a limitation in the report, since a
verbalized number does not imply the model is actually calibrated).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """One completion, with what's needed to compute latency and cost later."""

    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMDecision:
    """Mirrors `decision.Decision`, plus what's needed to compute cost later."""

    value: str
    confidence: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int


_CLASSIFY_PROMPT = """Classify the text below for: {question}

Respond with strict JSON only, no other text, in this exact shape:
{{"label": "<one of: {options}>", "confidence": <integer 0-100>}}

"confidence" is how confident you are that "label" is correct, expressed as
a percentage.

Text: {text}
"""


class LLMClient:
    def __init__(self, api_key: str, model: str, mock: bool = False) -> None:
        if not mock and not api_key:
            raise ValueError("api_key is required unless mock=True")
        self.model = model
        self.is_mock = mock
        self._client = None if mock else _build_anthropic_client(api_key)

    def complete(self, prompt: str, max_tokens: int = 256) -> LLMResponse:
        if self.is_mock:
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

    def classify(self, text: str, question: str, options: list[str]) -> LLMDecision:
        if self.is_mock:
            return _mock_classify(text, question, options)

        prompt = _CLASSIFY_PROMPT.format(question=question, options=", ".join(options), text=text)
        response = self.complete(prompt, max_tokens=50)
        label, confidence = _parse_classification(response.text, options)
        return LLMDecision(
            value=label,
            confidence=confidence,
            latency_seconds=response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


def _build_anthropic_client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def _parse_classification(text: str, options: list[str]) -> tuple[str, float]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in LLM response: {text!r}")
    payload = json.loads(match.group(0))
    label = payload["label"]
    if label not in options:
        raise ValueError(f"LLM returned label {label!r} outside options {options}")
    confidence = float(payload["confidence"]) / 100.0
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"LLM confidence {payload['confidence']!r} out of 0-100 range")
    return label, confidence


def _mock_complete(prompt: str) -> LLMResponse:
    """Deterministic stand-in: a hash of the prompt stands in for the model's text."""
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return LLMResponse(
        text=f"mock-response-{digest}",
        latency_seconds=0.0,
        input_tokens=len(prompt.split()),
        output_tokens=3,
    )


def _mock_classify(text: str, question: str, options: list[str]) -> LLMDecision:
    """Deterministic stand-in, mirroring `decision.MockDecisionClient.choice`."""
    digest = hashlib.sha256(f"{text}\x1f{question}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(options)
    return LLMDecision(
        value=options[index],
        confidence=0.7,
        latency_seconds=0.0,
        input_tokens=len(text.split()),
        output_tokens=5,
    )
