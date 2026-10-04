import pytest

from harness.metrics.accuracy import accuracy, accuracy_at_coverage


def test_accuracy_hand_computed() -> None:
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == pytest.approx(2 / 3)


def test_accuracy_all_correct() -> None:
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0


def test_accuracy_all_wrong() -> None:
    assert accuracy(["a", "b"], ["x", "y"]) == 0.0


def test_accuracy_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        accuracy([], [])


def test_accuracy_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        accuracy(["a"], ["a", "b"])


def test_accuracy_at_coverage_hand_computed_half() -> None:
    # Top 2 of 4 by confidence are indices 0, 1 (0.9, 0.8), both correct.
    confidences = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, False]
    result = accuracy_at_coverage(confidences, correct, target_coverage=0.5)
    assert result.n_answered == 2
    assert result.actual_coverage == 0.5
    assert result.confidence_threshold == 0.8
    assert result.accuracy == 1.0


def test_accuracy_at_coverage_full_coverage_is_plain_accuracy() -> None:
    confidences = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, False]
    result = accuracy_at_coverage(confidences, correct, target_coverage=1.0)
    assert result.n_answered == 4
    assert result.accuracy == 0.5


def test_accuracy_at_coverage_always_answers_at_least_one() -> None:
    confidences = [0.9, 0.1]
    correct = [True, False]
    result = accuracy_at_coverage(confidences, correct, target_coverage=0.1)
    assert result.n_answered == 1
    assert result.accuracy == 1.0


def test_accuracy_at_coverage_rejects_out_of_range_target() -> None:
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        accuracy_at_coverage([0.5], [True], target_coverage=0.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        accuracy_at_coverage([0.5], [True], target_coverage=1.5)
