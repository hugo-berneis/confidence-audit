#!/usr/bin/env python3
"""Prints split sizes and label balance for both tasks -- Phase 1's "done when".

Per CLAUDE.md's metric rules ("never report a number that wasn't produced by
a run"), a task that isn't ready yet (e.g. the 10-K gold sample isn't labeled
yet) is reported as skipped, not silently omitted or faked.

Usage:
    uv run scripts/describe_splits.py
"""

from __future__ import annotations

import pandas as pd

from harness.config import get_settings
from harness.data import risk_topics, sentiment
from harness.data.schema import SplitResult
from harness.data.splits import split_by_group, split_random


def _summarize(task: str, result: SplitResult) -> pd.DataFrame:
    rows = []
    for split_name, examples in (("tune", result.tune), ("test", result.test), ("ood", result.ood)):
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.label] = counts.get(example.label, 0) + 1
        if not counts:
            rows.append({"task": task, "split": split_name, "label": "(empty)", "n": 0})
            continue
        for label, n in sorted(counts.items()):
            rows.append({"task": task, "split": split_name, "label": label, "n": n})
    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()
    frames = []

    try:
        examples = risk_topics.load(settings.risk_topic_gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"risk_topic: skipped -- {exc}\n")
    else:
        result = split_by_group(
            examples, ood_groups=set(settings.risk_topic_ood_tickers), seed=settings.seed
        )
        frames.append(_summarize("risk_topic", result))

    sentiment_examples = sentiment.load()
    sentiment_result = split_random(sentiment_examples, seed=settings.seed)
    frames.append(_summarize("financial_sentiment", sentiment_result))

    if frames:
        print(pd.concat(frames, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
