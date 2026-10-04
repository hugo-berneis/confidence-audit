from dataclasses import dataclass

from harness.data.schema import Example
from harness.models.decision import MockDecisionClient
from harness.models.llm_client import LLMClient
from harness.models.predictors import jev_predict, llm_predict

EXAMPLE = Example(id="ex-1", text="Revenue grew 20% year over year.", label="positive", group="g")
OPTIONS = ["positive", "negative", "neutral"]


def test_jev_predict_returns_value_and_confidence_with_no_cost() -> None:
    predict = jev_predict(MockDecisionClient(), "sentiment?", OPTIONS)
    record = predict(EXAMPLE)
    assert record["prediction"] in OPTIONS
    assert record["confidence"] == 0.6
    assert record["latency_seconds"] is None
    assert record["cost_usd"] is None


def test_llm_predict_mock_has_no_cost() -> None:
    client = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)
    predict = llm_predict(client, "sentiment?", OPTIONS)
    record = predict(EXAMPLE)
    assert record["prediction"] in OPTIONS
    assert record["cost_usd"] is None
    assert record["latency_seconds"] == 0.0


@dataclass
class _FakeDecision:
    value: str
    confidence: float
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class _FakeRealLLMClient:
    """Duck-types the bits of LLMClient that llm_predict() uses, without a network call."""

    is_mock = False
    model = "claude-haiku-4-5-20251001"

    def classify(self, text: str, question: str, options: list[str]) -> _FakeDecision:
        return _FakeDecision(
            value=options[0], confidence=0.9, latency_seconds=1.2, input_tokens=10, output_tokens=5
        )


def test_llm_predict_real_computes_cost_from_pricing_table() -> None:
    predict = llm_predict(_FakeRealLLMClient(), "sentiment?", OPTIONS)
    record = predict(EXAMPLE)
    assert record["cost_usd"] == (10 * 1.00 + 5 * 5.00) / 1_000_000
    assert record["latency_seconds"] == 1.2
