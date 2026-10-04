"""Generic model-runner loop.

Predicts on every example not already in the cache, retrying transient
failures with backoff before giving up on an example. The cache is the
single source of truth for what's been run -- a rerun always resumes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.data.schema import Example
from harness.models.cache import PredictionCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunStats:
    n_total: int
    n_cached: int
    n_run: int
    failed_ids: list[str] = field(default_factory=list)


def run_predictions(
    task: str,
    model: str,
    examples: list[Example],
    predict: Callable[[Example], dict[str, Any]],
    cache: PredictionCache,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> RunStats:
    n_cached = 0
    n_run = 0
    failed_ids: list[str] = []
    for example in examples:
        if cache.get(task, model, example.id) is not None:
            n_cached += 1
            continue
        record = _predict_with_retry(predict, example, max_retries, backoff_seconds)
        if record is None:
            failed_ids.append(example.id)
            continue
        cache.put(task, model, example.id, record)
        n_run += 1
    return RunStats(n_total=len(examples), n_cached=n_cached, n_run=n_run, failed_ids=failed_ids)


def _predict_with_retry(
    predict: Callable[[Example], dict[str, Any]],
    example: Example,
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any] | None:
    for attempt in range(max_retries + 1):
        try:
            return predict(example)
        except Exception:
            if attempt == max_retries:
                logger.warning(
                    "giving up on %s after %d attempts", example.id, attempt + 1, exc_info=True
                )
                return None
            time.sleep(backoff_seconds * (2**attempt))
    return None
