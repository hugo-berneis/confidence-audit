from pathlib import Path

from harness.models.cache import PredictionCache


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "predictions.jsonl")
    assert cache.get("task", "model", "example-1") is None


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "predictions.jsonl")
    cache.put("task", "model", "example-1", {"label": "A", "confidence": 0.8})
    record = cache.get("task", "model", "example-1")
    assert record is not None
    assert record["label"] == "A"
    assert record["confidence"] == 0.8


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    PredictionCache(path).put("task", "model", "example-1", {"label": "A"})
    reloaded = PredictionCache(path)
    assert reloaded.get("task", "model", "example-1") == {
        "task": "task",
        "model": "model",
        "example_id": "example-1",
        "label": "A",
    }


def test_rewriting_a_key_keeps_latest_value(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    cache = PredictionCache(path)
    cache.put("task", "model", "example-1", {"label": "A"})
    cache.put("task", "model", "example-1", {"label": "B"})
    assert cache.get("task", "model", "example-1")["label"] == "B"
    assert len(PredictionCache(path)) == 1
