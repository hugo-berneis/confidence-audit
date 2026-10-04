from pathlib import Path

from harness.data.schema import Example
from harness.models.cache import PredictionCache
from harness.models.runner import run_predictions

EXAMPLES = [Example(id="ex-1", text="a", label="x", group="g")]


def test_runs_and_caches_a_new_example(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "preds.jsonl")
    calls = []

    def predict(example: Example) -> dict:
        calls.append(example.id)
        return {"prediction": "x", "confidence": 0.9}

    stats = run_predictions("task", "model", EXAMPLES, predict, cache)

    assert stats.n_run == 1
    assert stats.n_cached == 0
    assert calls == ["ex-1"]
    assert cache.get("task", "model", "ex-1")["prediction"] == "x"


def test_skips_examples_already_in_cache(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "preds.jsonl")
    cache.put("task", "model", "ex-1", {"prediction": "x", "confidence": 0.9})
    calls = []

    def predict(example: Example) -> dict:
        calls.append(example.id)
        return {"prediction": "should-not-happen"}

    stats = run_predictions("task", "model", EXAMPLES, predict, cache)

    assert stats.n_cached == 1
    assert stats.n_run == 0
    assert calls == []


def test_retries_then_succeeds(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "preds.jsonl")
    attempts = {"n": 0}

    def predict(example: Example) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return {"prediction": "x"}

    stats = run_predictions("task", "model", EXAMPLES, predict, cache, backoff_seconds=0)

    assert attempts["n"] == 3
    assert stats.n_run == 1
    assert stats.failed_ids == []


def test_gives_up_after_max_retries_and_does_not_cache(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path / "preds.jsonl")

    def predict(example: Example) -> dict:
        raise RuntimeError("always fails")

    stats = run_predictions(
        "task", "model", EXAMPLES, predict, cache, max_retries=2, backoff_seconds=0
    )

    assert stats.n_run == 0
    assert stats.failed_ids == ["ex-1"]
    assert cache.get("task", "model", "ex-1") is None


def test_rerun_resumes_only_the_failed_example(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    cache = PredictionCache(path)
    examples = [
        Example(id="ex-1", text="a", label="x", group="g"),
        Example(id="ex-2", text="b", label="y", group="g"),
    ]

    def flaky_predict(example: Example) -> dict:
        if example.id == "ex-2":
            raise RuntimeError("down")
        return {"prediction": "x"}

    first = run_predictions("task", "model", examples, flaky_predict, cache, backoff_seconds=0)
    assert first.failed_ids == ["ex-2"]

    second_cache = PredictionCache(path)
    calls = []

    def recovered_predict(example: Example) -> dict:
        calls.append(example.id)
        return {"prediction": "y"}

    second = run_predictions(
        "task", "model", examples, recovered_predict, second_cache, backoff_seconds=0
    )

    assert calls == ["ex-2"]
    assert second.n_cached == 1
    assert second.n_run == 1
