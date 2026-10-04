#!/usr/bin/env python3
"""Selects tau_low/tau_high on the tune split only, freezes them, and
applies them unchanged to test (and OOD, once the 10-K task has gold
labels). Phase 4's "done when": `exports/thresholds.json` validates against
`configs/thresholds.schema.json`, and the tune/test/OOD comparison table
exists (printed here, and written alongside as `thresholds_report.json`).

Usage:
    uv run scripts/export_thresholds.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.config import Settings, get_settings
from harness.data import risk_topics, sentiment
from harness.data.schema import Example, SplitResult
from harness.data.splits import split_by_group, split_random
from harness.models.cache import PredictionCache
from harness.thresholds.schema import validate_thresholds_export
from harness.thresholds.select import apply_thresholds, select_thresholds

EXPORT_PATH = Path("exports/thresholds.json")
REPORT_PATH = Path("exports/thresholds_report.json")


def _confidences_and_correctness(
    examples_by_id: dict[str, Example], records: list[dict[str, Any]]
) -> tuple[list[float], list[bool]]:
    confidences = [r["confidence"] for r in records]
    correct = [r["prediction"] == examples_by_id[r["example_id"]].label for r in records]
    return confidences, correct


def _records_for(
    cache: PredictionCache, task: str, model: str, split_ids: set[str]
) -> list[dict[str, Any]]:
    return [
        r
        for r in cache.entries_for_task(task)
        if r["model"] == model and r["example_id"] in split_ids
    ]


def _process_task(
    task: str,
    examples: list[Example],
    split_result: SplitResult,
    cache: PredictionCache,
    settings: Settings,
    run_id: str,
    today: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples_by_id = {e.id: e for e in examples}
    split_ids = {
        "tune": {e.id for e in split_result.tune},
        "test": {e.id for e in split_result.test},
        "ood": {e.id for e in split_result.ood},
    }
    models = sorted({r["model"] for r in cache.entries_for_task(task)})

    export_records = []
    report: dict[str, Any] = {}

    for model in models:
        tune_records = _records_for(cache, task, model, split_ids["tune"])
        if not tune_records:
            continue
        confidences, correct = _confidences_and_correctness(examples_by_id, tune_records)
        thresholds = select_thresholds(
            confidences, correct, settings.target_coverage, settings.target_error_rate
        )

        export_records.append(
            {
                "task": task,
                "model": model,
                "tau_low": thresholds.tau_low,
                "tau_high": thresholds.tau_high,
                "target": {
                    "coverage": thresholds.target_coverage,
                    "error_rate": thresholds.target_error_rate,
                },
                "date": today,
                "run_id": run_id,
            }
        )

        split_report = {}
        for split_name, ids in split_ids.items():
            split_records = _records_for(cache, task, model, ids)
            if not split_records:
                continue
            split_confidences, split_correct = _confidences_and_correctness(
                examples_by_id, split_records
            )
            applied = apply_thresholds(thresholds, split_confidences, split_correct)
            split_report[split_name] = asdict(applied)
        report[model] = split_report

    return export_records, report


def _print_comparison_table(all_report: dict[str, Any]) -> None:
    header = (
        f"{'task':<22}{'model':<12}{'split':<8}{'coverage':>10}{'accuracy':>10}{'trusted_acc':>13}"
    )
    print(header)
    print("-" * len(header))
    for task, by_model in all_report.items():
        for model, by_split in by_model.items():
            for split, applied in by_split.items():
                trusted_acc = applied["trusted_accuracy"]
                trusted_str = f"{trusted_acc:.3f}" if trusted_acc is not None else "n/a"
                print(
                    f"{task:<22}{model:<12}{split:<8}"
                    f"{applied['answered_coverage']:>10.3f}"
                    f"{applied['answered_accuracy']:>10.3f}"
                    f"{trusted_str:>13}"
                )


def main() -> None:
    settings = get_settings()
    cache = PredictionCache(Path(settings.cache_dir) / "predictions.jsonl")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    today = datetime.now(UTC).date().isoformat()

    all_export_records: list[dict[str, Any]] = []
    all_report: dict[str, Any] = {}

    try:
        risk_examples = risk_topics.load(settings.risk_topic_gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"risk_topic: skipped -- {exc}\n")
    else:
        result = split_by_group(
            risk_examples, ood_groups=set(settings.risk_topic_ood_tickers), seed=settings.seed
        )
        records, report = _process_task(
            risk_topics.TASK, risk_examples, result, cache, settings, run_id, today
        )
        all_export_records.extend(records)
        all_report[risk_topics.TASK] = report

    sentiment_examples = sentiment.load()
    sentiment_result = split_random(sentiment_examples, seed=settings.seed)
    records, report = _process_task(
        sentiment.TASK, sentiment_examples, sentiment_result, cache, settings, run_id, today
    )
    all_export_records.extend(records)
    all_report[sentiment.TASK] = report

    if not all_export_records:
        raise SystemExit("No cached predictions yet -- run scripts/run_models.py first.")

    validate_thresholds_export(all_export_records)

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(all_export_records, indent=2))
    REPORT_PATH.write_text(json.dumps(all_report, indent=2))

    print(f"Wrote {EXPORT_PATH} ({len(all_export_records)} threshold records, schema-validated)")
    print(f"Wrote {REPORT_PATH}\n")
    _print_comparison_table(all_report)


if __name__ == "__main__":
    main()
