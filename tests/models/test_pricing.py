import pytest

from harness.models.pricing import cost_usd


def test_known_model_computes_cost() -> None:
    cost = cost_usd("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(1.00 + 5.00)


def test_zero_tokens_cost_zero() -> None:
    assert cost_usd("claude-haiku-4-5-20251001", 0, 0) == 0.0


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="no pricing on file"):
        cost_usd("some-other-model", 100, 100)
