import pytest

from harness.metrics.discrimination import auroc_confidence_vs_correctness


def test_auroc_perfect_separation() -> None:
    confidences = [0.9, 0.8, 0.3, 0.2]
    correct = [True, True, False, False]
    assert auroc_confidence_vs_correctness(confidences, correct) == pytest.approx(1.0)


def test_auroc_inverse_separation() -> None:
    confidences = [0.9, 0.8, 0.3, 0.2]
    correct = [False, False, True, True]
    assert auroc_confidence_vs_correctness(confidences, correct) == pytest.approx(0.0)


def test_auroc_no_discrimination_is_half() -> None:
    confidences = [0.5, 0.5, 0.5, 0.5]
    correct = [True, False, True, False]
    assert auroc_confidence_vs_correctness(confidences, correct) == pytest.approx(0.5)


def test_auroc_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="undefined"):
        auroc_confidence_vs_correctness([0.9, 0.8], [True, True])


def test_auroc_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        auroc_confidence_vs_correctness([0.9], [True, False])
