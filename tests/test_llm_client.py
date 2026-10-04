import pytest

from harness.models.llm_client import LLMClient


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
