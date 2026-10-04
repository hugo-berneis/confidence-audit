import pytest

from harness.models.llm_client import LLMClient, _parse_classification


def test_mock_complete_is_deterministic() -> None:
    client = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)
    first = client.complete("classify this paragraph")
    second = client.complete("classify this paragraph")
    assert first == second


def test_mock_complete_varies_with_prompt() -> None:
    client = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)
    a = client.complete("prompt A")
    b = client.complete("prompt B")
    assert a.text != b.text


def test_real_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=False)


def test_mock_classify_returns_an_option() -> None:
    client = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)
    decision = client.classify("Revenue grew 20%.", "sentiment?", ["positive", "negative"])
    assert decision.value in {"positive", "negative"}
    assert decision.confidence == 0.7


def test_mock_classify_is_deterministic() -> None:
    client = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)
    options = ["positive", "negative", "neutral"]
    first = client.classify("same text", "sentiment?", options)
    second = client.classify("same text", "sentiment?", options)
    assert first == second


def test_parse_classification_extracts_label_and_confidence() -> None:
    label, confidence = _parse_classification(
        '{"label": "positive", "confidence": 80}', ["positive", "negative"]
    )
    assert label == "positive"
    assert confidence == 0.8


def test_parse_classification_tolerates_surrounding_text() -> None:
    label, confidence = _parse_classification(
        'Sure, here you go: {"label": "negative", "confidence": 50} thanks!',
        ["positive", "negative"],
    )
    assert label == "negative"
    assert confidence == 0.5


def test_parse_classification_rejects_label_outside_options() -> None:
    with pytest.raises(ValueError, match="outside options"):
        _parse_classification('{"label": "sideways", "confidence": 90}', ["positive", "negative"])


def test_parse_classification_rejects_missing_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _parse_classification("I am not confident enough to answer.", ["positive", "negative"])


def test_parse_classification_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of 0-100 range"):
        _parse_classification('{"label": "positive", "confidence": 150}', ["positive", "negative"])
