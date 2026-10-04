#!/usr/bin/env python3
"""Runs both models (Jev, the LLM) on every split of each task through the
prediction cache. Mock clients only -- Phase 2's "done when" is the mock
prediction files existing for both models and both tasks; real-client runs
cost money and need `.env` API keys, so they're a separate, explicit step
(see README) rather than something this script does on its own.

Per CLAUDE.md's metric rules, a task that isn't ready (e.g. the 10-K gold
sample isn't labeled yet) is reported as skipped, not silently omitted.

Usage:
    uv run scripts/run_models.py
"""

from __future__ import annotations

from pathlib import Path

from harness.config import get_settings
from harness.data import risk_topics, sentiment
from harness.data.schema import Example
from harness.data.splits import split_by_group, split_random
from harness.models.cache import PredictionCache
from harness.models.decision import MockDecisionClient
from harness.models.llm_client import LLMClient
from harness.models.predictors import (
    RISK_TOPIC_QUESTION,
    SENTIMENT_OPTIONS,
    SENTIMENT_QUESTION,
    jev_predict,
    llm_predict,
)
from harness.models.runner import run_predictions


def _run_task(
    task: str,
    examples: list[Example],
    question: str,
    options: list[str],
    cache: PredictionCache,
    llm_model: str,
) -> None:
    jev = MockDecisionClient()
    llm = LLMClient(api_key="", model=llm_model, mock=True)

    jev_stats = run_predictions(
        task, "jev-mock", examples, jev_predict(jev, question, options), cache
    )
    llm_stats = run_predictions(
        task, "llm-mock", examples, llm_predict(llm, question, options), cache
    )

    print(
        f"{task}: jev-mock -> ran {jev_stats.n_run}, cached {jev_stats.n_cached}, "
        f"failed {len(jev_stats.failed_ids)}"
    )
    print(
        f"{task}: llm-mock -> ran {llm_stats.n_run}, cached {llm_stats.n_cached}, "
        f"failed {len(llm_stats.failed_ids)}"
    )


def main() -> None:
    settings = get_settings()
    cache = PredictionCache(Path(settings.cache_dir) / "predictions.jsonl")

    try:
        risk_examples = risk_topics.load(settings.risk_topic_gold_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"risk_topic: skipped -- {exc}\n")
    else:
        result = split_by_group(
            risk_examples, ood_groups=set(settings.risk_topic_ood_tickers), seed=settings.seed
        )
        all_examples = result.tune + result.test + result.ood
        _run_task(
            risk_topics.TASK,
            all_examples,
            RISK_TOPIC_QUESTION,
            risk_topics.TOPICS,
            cache,
            settings.anthropic_model,
        )

    sentiment_examples = sentiment.load()
    result = split_random(sentiment_examples, seed=settings.seed)
    all_examples = result.tune + result.test
    _run_task(
        sentiment.TASK,
        all_examples,
        SENTIMENT_QUESTION,
        SENTIMENT_OPTIONS,
        cache,
        settings.anthropic_model,
    )


if __name__ == "__main__":
    main()
