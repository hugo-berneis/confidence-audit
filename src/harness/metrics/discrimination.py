"""AUROC of confidence as a predictor of correctness.

Uses scikit-learn for this one metric, per CLAUDE.md's metric rules --
everything else in this package is hand-written and hand-tested.
"""

from __future__ import annotations


def auroc_confidence_vs_correctness(confidences: list[float], correct: list[bool]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if len(set(correct)) < 2:
        raise ValueError(
            "AUROC is undefined when every prediction is correct or every prediction is wrong"
        )
    return float(roc_auc_score(correct, confidences))
