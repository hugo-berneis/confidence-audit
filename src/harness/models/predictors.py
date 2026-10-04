"""Task-specific predict functions, wiring a `DecisionClient` or `LLMClient`
into the generic runner (`runner.run_predictions`). Each returns a plain
dict ready for `PredictionCache.put`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.data.schema import Example
from harness.models.decision import DecisionClient
from harness.models.llm_client import LLMClient
from harness.models.pricing import cost_usd

RISK_TOPIC_QUESTION = "What risk topic does this paragraph primarily discuss?"
SENTIMENT_QUESTION = "What is the sentiment of this financial news text?"
SENTIMENT_OPTIONS = ["negative", "neutral", "positive"]


def jev_predict(
    client: DecisionClient, question: str, options: list[str]
) -> Callable[[Example], dict[str, Any]]:
    def predict(example: Example) -> dict[str, Any]:
        decision = client.choice(example.text, question, options)
        return {
            "prediction": decision.value,
            "confidence": decision.confidence,
            # Real Jev's per-call latency/cost isn't wired up yet -- Phase 0
            # scoped the real JevDecisionClient out; see decision.py.
            "latency_seconds": None,
            "cost_usd": None,
        }

    return predict


def llm_predict(
    client: LLMClient, question: str, options: list[str]
) -> Callable[[Example], dict[str, Any]]:
    def predict(example: Example) -> dict[str, Any]:
        decision = client.classify(example.text, question, options)
        cost = (
            None
            if client.is_mock
            else cost_usd(client.model, decision.input_tokens, decision.output_tokens)
        )
        return {
            "prediction": decision.value,
            "confidence": decision.confidence,
            "latency_seconds": decision.latency_seconds,
            "cost_usd": cost,
            "input_tokens": decision.input_tokens,
            "output_tokens": decision.output_tokens,
        }

    return predict
