#!/usr/bin/env python3
"""Computes accuracy, ECE, AUROC, and accuracy-at-coverage for every
(task, model, split) in the prediction cache, and writes reliability
diagrams. Phase 3's "done when": `runs/<run_id>/metrics.json` plus the
chart PNGs.

Per CLAUDE.md's metric rules, AUROC is undefined when every cached
prediction for a group is correct or every one is wrong (no two classes to
separate) -- that's reported as `null` with an explanatory error message
rather than crashing or silently omitting the group.

Usage:
    uv run scripts/evaluate.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.config import get_settings
from harness.data import risk_topics, sentiment
from harness.data.schema import Example, SplitResult
from harness.data.splits import split_by_group, split_random
from harness.metrics.accuracy import accuracy, accuracy_at_coverage
from harness.metrics.calibration import expected_calibration_error, reliability_bins
from harness.metrics.discrimination import auroc_confidence_vs_correctness
from harness.models.cache import PredictionCache
from harness.report.charts import plot_reliability_diagram

COVERAGE_TARGETS = (0.8, 0.9)


def _split_lookup(split_result: SplitResult) -> dict[str, str]:
    lookup = {}
    for name, examples in (
        ("tune", split_result.tune),
        ("test", split_result.test),
        ("ood", split_result.ood),
    ):
        for example in examples:
            lookup[example.id] = name
    return lookup


def _evaluate_group(
    task: str,
    model: str,
    split: str,
    examples_by_id: dict[str, Example],
    records: list[dict[str, Any]],
    charts_dir: Path,
) -> dict[str, Any]:
    predictions = [r["prediction"] for r in records]
    labels = [examples_by_id[r["example_id"]].label for r in records]
    confidences = [r["confidence"] for r in records]
    correct = [p == label for p, label in zip(predictions, labels, strict=True)]

    metrics: dict[str, Any] = {
        "n": len(records),
        "accuracy": accuracy(predictions, labels),
        "ece": expected_calibration_error(confidences, correct),
    }

    try:
        metrics["auroc"] = auroc_confidence_vs_correctness(confidences, correct)
    except ValueError as exc:
        metrics["auroc"] = None
        metrics["auroc_error"] = str(exc)

    metrics["accuracy_at_coverage"] = {
        target: asdict(accuracy_at_coverage(confidences, correct, target))
        for target in COVERAGE_TARGETS
    }

    chart_path = charts_dir / f"{task}_{model}_{split}_reliability.png"
    plot_reliability_diagram(
        reliability_bins(confidences, correct),
        title=f"{task} / {model} / {split}",
        output_path=chart_path,
    )
    metrics["reliability_chart"] = str(chart_path)
    return metrics


def _evaluate_task(
    task: str,
    examples: list[Example],
    split_result: SplitResult,
    cache: PredictionCache,
    charts_dir: Path,
) -> dict[str, Any]:
    examples_by_id = {e.id: e for e in examples}
    split_lookup = _split_lookup(split_result)

    by_model_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in cache.entries_for_task(task):
        split = split_lookup.get(record["example_id"])
        if split is None:
            continue  # stale cache row from a prior split/seed -- don't let it break the run
        by_model_split[(record["model"], split)].append(record)

    result: dict[str, Any] = {}
    for (model, split), records in sorted(by_model_split.items()):
        result.setdefault(model, {})[split] = _evaluate_group(
            task, model, split, examples_by_id, records, charts_dir
        )
    return result


def main() -> None:
    settings = get_settings()
    cache = PredictionCache(Path(settings.cache_dir) / "predictions.jsonl")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("runs") / run_id
    charts_dir = run_dir / "charts"

    all_metrics: dict[str, Any] = {}

    try:
        risk_examples = risk_topics.load(settings.risk_topic_gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"risk_topic: skipped -- {exc}\n")
    else:
        result = split_by_group(
            risk_examples, ood_groups=set(settings.risk_topic_ood_tickers), seed=settings.seed
        )
        all_metrics[risk_topics.TASK] = _evaluate_task(
            risk_topics.TASK, risk_examples, result, cache, charts_dir
        )

    sentiment_examples = sentiment.load()
    sentiment_result = split_random(sentiment_examples, seed=settings.seed)
    all_metrics[sentiment.TASK] = _evaluate_task(
        sentiment.TASK, sentiment_examples, sentiment_result, cache, charts_dir
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(settings.model_dump(exclude={"anthropic_api_key", "jev_api_key"}), indent=2)
    )
    print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
