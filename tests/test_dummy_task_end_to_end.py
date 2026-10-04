"""Phase 0 'done when': a dummy classification task runs end-to-end on the
mocks, through the cache, for both models.
"""

from pathlib import Path

from harness.models.cache import PredictionCache
from harness.models.decision import MockDecisionClient
from harness.models.llm_client import LLMClient

DUMMY_TASK = "dummy_sentiment"
LABELS = ["positive", "neutral", "negative"]
DUMMY_EXAMPLES = [
    {"id": "ex-1", "text": "Revenue grew 20% year over year."},
    {"id": "ex-2", "text": "The company missed earnings estimates."},
    {"id": "ex-3", "text": "No material change from last quarter."},
]


def test_dummy_task_runs_end_to_end_on_mocks(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "predictions.jsonl")
    jev = MockDecisionClient()
    llm = LLMClient(api_key="", model="claude-haiku-4-5-20251001", mock=True)

    for example in DUMMY_EXAMPLES:
        decision = jev.choice(example["text"], "What is the sentiment?", LABELS)
        cache.put(
            DUMMY_TASK,
            "jev-mock",
            example["id"],
            {"prediction": decision.value, "confidence": decision.confidence},
        )

        response = llm.complete(f"Classify the sentiment ({LABELS}) of: {example['text']}")
        cache.put(
            DUMMY_TASK,
            "llm-mock",
            example["id"],
            {"prediction": response.text, "latency_seconds": response.latency_seconds},
        )

    assert len(cache) == len(DUMMY_EXAMPLES) * 2
    for example in DUMMY_EXAMPLES:
        assert cache.get(DUMMY_TASK, "jev-mock", example["id"])["prediction"] in LABELS
        assert cache.get(DUMMY_TASK, "llm-mock", example["id"]) is not None


def test_rerun_resumes_from_cache_without_recomputing(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    jev = MockDecisionClient()

    first_run = PredictionCache(path)
    decision = jev.choice(DUMMY_EXAMPLES[0]["text"], "What is the sentiment?", LABELS)
    first_run.put(DUMMY_TASK, "jev-mock", "ex-1", {"prediction": decision.value})

    second_run = PredictionCache(path)
    cached = second_run.get(DUMMY_TASK, "jev-mock", "ex-1")
    assert cached is not None
    assert cached["prediction"] == decision.value
