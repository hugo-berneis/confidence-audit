"""Calibration metrics for a scalar confidence predicting correctness.

Each prediction is treated as a Bernoulli trial: confidence is the model's
stated probability that it's right, and "correct" (prediction == gold label)
is the outcome. ECE and the reliability diagram are the standard tools for
checking whether that stated probability matches the empirical rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> list[ReliabilityBin]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    bucketed: list[list[int]] = [[] for _ in range(n_bins)]
    for i, conf in enumerate(confidences):
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence {conf} out of [0, 1]")
        index = min(int(conf * n_bins), n_bins - 1)
        bucketed[index].append(i)

    bins = []
    for b, indices in enumerate(bucketed):
        lower, upper = b / n_bins, (b + 1) / n_bins
        if not indices:
            bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
            continue
        mean_conf = sum(confidences[i] for i in indices) / len(indices)
        acc = sum(correct[i] for i in indices) / len(indices)
        bins.append(ReliabilityBin(lower, upper, len(indices), mean_conf, acc))
    return bins


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    bins = reliability_bins(confidences, correct, n_bins)
    n_total = len(confidences)
    return sum(b.count / n_total * abs(b.accuracy - b.mean_confidence) for b in bins if b.count)
