import pytest

from harness.metrics.calibration import expected_calibration_error, reliability_bins


def test_reliability_bins_hand_computed() -> None:
    # 0.9s land in bin 9 ([0.9, 1.0)); 0.6s land in bin 6 ([0.6, 0.7)).
    confidences = [0.9, 0.9, 0.6, 0.6]
    correct = [True, True, True, False]
    bins = reliability_bins(confidences, correct, n_bins=10)

    assert len(bins) == 10
    assert bins[9].count == 2
    assert bins[9].mean_confidence == pytest.approx(0.9)
    assert bins[9].accuracy == pytest.approx(1.0)
    assert bins[6].count == 2
    assert bins[6].mean_confidence == pytest.approx(0.6)
    assert bins[6].accuracy == pytest.approx(0.5)
    # every other bin is empty
    assert sum(b.count for i, b in enumerate(bins) if i not in (6, 9)) == 0


def test_reliability_bins_confidence_of_one_lands_in_last_bin() -> None:
    bins = reliability_bins([1.0], [True], n_bins=2)
    assert bins[1].count == 1
    assert bins[0].count == 0


def test_reliability_bins_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="out of \\[0, 1\\]"):
        reliability_bins([1.5], [True])


def test_expected_calibration_error_hand_computed() -> None:
    # Bin 9: |1.0 - 0.9| * (2/4) = 0.05. Bin 6: |0.5 - 0.6| * (2/4) = 0.05. Total 0.1.
    confidences = [0.9, 0.9, 0.6, 0.6]
    correct = [True, True, True, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.1)


def test_expected_calibration_error_is_zero_when_perfectly_calibrated() -> None:
    # Every confidence matches the empirical accuracy in its own bin (0.0 or 1.0, trivially).
    confidences = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) == pytest.approx(0.0)
