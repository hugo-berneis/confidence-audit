"""Accuracy-based metrics: plain accuracy and accuracy at a target coverage.

Coverage is the fraction of examples the model is allowed to answer, kept by
confidence rank -- the rest are treated as abstentions. This is the standard
selective-prediction framing: a model that abstains on its least confident
examples should look better than one forced to answer everything.
"""

from __future__ import annotations

from dataclasses import dataclass


def accuracy(predictions: list[str], labels: list[str]) -> float:
    if not predictions:
        raise ValueError("accuracy() requires at least one example")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must be the same length")
    n_correct = sum(p == label for p, label in zip(predictions, labels, strict=True))
    return n_correct / len(predictions)


@dataclass(frozen=True)
class CoverageResult:
    target_coverage: float
    n_answered: int
    n_total: int
    actual_coverage: float
    confidence_threshold: float
    accuracy: float


def accuracy_at_coverage(
    confidences: list[float], correct: list[bool], target_coverage: float
) -> CoverageResult:
    """Accuracy among the `target_coverage` fraction of examples with the highest confidence."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("accuracy_at_coverage() requires at least one example")

    n_total = len(confidences)
    n_answered = max(1, round(target_coverage * n_total))
    order = sorted(range(n_total), key=lambda i: confidences[i], reverse=True)
    answered = order[:n_answered]
    acc = sum(correct[i] for i in answered) / n_answered
    return CoverageResult(
        target_coverage=target_coverage,
        n_answered=n_answered,
        n_total=n_total,
        actual_coverage=n_answered / n_total,
        confidence_threshold=confidences[answered[-1]],
        accuracy=acc,
    )
