"""Threshold selection: pick tau_low and tau_high on the tune split only,
then freeze them before touching test or OOD.

Mirrors Project 1's existing never-silently-drop-on-uncertainty philosophy
(`config/thresholds.yaml` there has the same accept-above / reject-below
shape, just as two independent gates):

- tau_high: the smallest confidence that keeps the error rate among
  accepted ("trusted") predictions at or under `target_error_rate`.
- tau_low: the confidence needed to answer at least `target_coverage` of
  examples outright (reject anything below it).
- Between tau_low and tau_high: still answered, but flagged low-confidence.

Confirmed with Hugo: both thresholds are tuned on the tune split only;
test and OOD only ever have these frozen values applied to them.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.metrics.accuracy import accuracy_at_coverage


@dataclass(frozen=True)
class Thresholds:
    tau_low: float
    tau_high: float
    target_coverage: float
    target_error_rate: float


@dataclass(frozen=True)
class AppliedThresholds:
    """What a frozen (tau_low, tau_high) pair looks like applied to a split."""

    n_total: int
    n_rejected: int
    n_warned: int
    n_trusted: int
    rejected_fraction: float
    warned_fraction: float
    trusted_fraction: float
    answered_coverage: float
    answered_accuracy: float
    trusted_accuracy: float | None


def select_thresholds(
    confidences: list[float],
    correct: list[bool],
    target_coverage: float,
    target_error_rate: float,
) -> Thresholds:
    if not confidences:
        raise ValueError("select_thresholds() requires at least one example")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")

    tau_low = accuracy_at_coverage(confidences, correct, target_coverage).confidence_threshold
    # Restricted to >= tau_low so "trusted" is always a subset of "answered" --
    # otherwise an independent search could return tau_high < tau_low.
    tau_high = _smallest_threshold_meeting_error_rate(
        confidences, correct, target_error_rate, lower_bound=tau_low
    )

    return Thresholds(
        tau_low=tau_low,
        tau_high=tau_high,
        target_coverage=target_coverage,
        target_error_rate=target_error_rate,
    )


def apply_thresholds(
    thresholds: Thresholds, confidences: list[float], correct: list[bool]
) -> AppliedThresholds:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("apply_thresholds() requires at least one example")

    n_total = len(confidences)
    pairs = list(zip(correct, confidences, strict=True))
    rejected = [c for c, conf in pairs if conf < thresholds.tau_low]
    trusted = [c for c, conf in pairs if conf >= thresholds.tau_high]
    answered = [c for c, conf in pairs if conf >= thresholds.tau_low]
    n_warned = len(answered) - len(trusted)

    return AppliedThresholds(
        n_total=n_total,
        n_rejected=len(rejected),
        n_warned=n_warned,
        n_trusted=len(trusted),
        rejected_fraction=len(rejected) / n_total,
        warned_fraction=n_warned / n_total,
        trusted_fraction=len(trusted) / n_total,
        answered_coverage=len(answered) / n_total,
        answered_accuracy=(sum(answered) / len(answered)) if answered else 0.0,
        trusted_accuracy=(sum(trusted) / len(trusted)) if trusted else None,
    )


def _smallest_threshold_meeting_error_rate(
    confidences: list[float],
    correct: list[bool],
    target_error_rate: float,
    lower_bound: float = 0.0,
) -> float:
    """Smallest confidence threshold `t` (>= `lower_bound`) such that
    accepting only confidence >= t keeps the error rate (1 - accuracy) at or
    under `target_error_rate`.

    Candidates are tried from lowest to highest, so ties favor coverage --
    answering more examples is preferred whenever the target is still met.
    If no threshold meets the target, even accepting only the single most
    confident example, falls back to the highest observed confidence (the
    strictest band the data supports) -- a threshold still has to be
    exported, but `apply_thresholds()` reports the real achieved error rate
    at that threshold on every split, so nothing is hidden.
    """
    candidates = sorted(c for c in set(confidences) if c >= lower_bound)
    for t in candidates:
        accepted = [c for c, conf in zip(correct, confidences, strict=True) if conf >= t]
        if not accepted:
            continue
        error_rate = 1 - sum(accepted) / len(accepted)
        if error_rate <= target_error_rate:
            return t
    return max(confidences)
