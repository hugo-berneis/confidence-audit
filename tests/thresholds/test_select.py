import pytest

from harness.thresholds.select import Thresholds, apply_thresholds, select_thresholds

CONFIDENCES = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
CORRECT = [True, False, True, True, True, False]


def test_select_thresholds_hand_computed() -> None:
    # tau_low: top 50% by confidence (3 of 6) are [0.95, 0.9, 0.85] -> threshold 0.85.
    # tau_high: smallest t >= 0.85 with zero error among accepted -- only t=0.95
    # (accepting [0.95] alone, which is correct) has error 0; t=0.85 and t=0.9
    # both include the one wrong (0.9, False), giving error > 0.
    result = select_thresholds(
        CONFIDENCES, CORRECT, target_coverage=0.5, target_error_rate=0.0
    )
    assert result.tau_low == pytest.approx(0.85)
    assert result.tau_high == pytest.approx(0.95)


def test_tau_high_is_never_below_tau_low() -> None:
    # A lenient error target would otherwise let tau_high collapse below tau_low.
    result = select_thresholds(
        CONFIDENCES, CORRECT, target_coverage=0.5, target_error_rate=0.9
    )
    assert result.tau_high >= result.tau_low


def test_select_thresholds_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        select_thresholds([], [], target_coverage=0.5, target_error_rate=0.1)


def test_select_thresholds_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        select_thresholds([0.5], [True, False], target_coverage=0.5, target_error_rate=0.1)


def test_apply_thresholds_hand_computed() -> None:
    thresholds = Thresholds(
        tau_low=0.85, tau_high=0.95, target_coverage=0.5, target_error_rate=0.0
    )
    result = apply_thresholds(thresholds, CONFIDENCES, CORRECT)

    assert result.n_total == 6
    assert result.n_rejected == 3  # confidences 0.8, 0.7, 0.6
    assert result.n_trusted == 1  # confidence 0.95
    assert result.n_warned == 2  # confidences 0.9, 0.85
    assert result.rejected_fraction == pytest.approx(0.5)
    assert result.trusted_fraction == pytest.approx(1 / 6)
    assert result.warned_fraction == pytest.approx(2 / 6)
    assert result.answered_coverage == pytest.approx(0.5)
    assert result.answered_accuracy == pytest.approx(2 / 3)
    assert result.trusted_accuracy == pytest.approx(1.0)


def test_apply_thresholds_trusted_accuracy_is_none_when_nothing_is_trusted() -> None:
    thresholds = Thresholds(tau_low=0.0, tau_high=1.1, target_coverage=1.0, target_error_rate=0.0)
    result = apply_thresholds(thresholds, [0.5, 0.6], [True, False])
    assert result.n_trusted == 0
    assert result.trusted_accuracy is None


def test_apply_thresholds_rejects_length_mismatch() -> None:
    thresholds = Thresholds(tau_low=0.5, tau_high=0.9, target_coverage=0.5, target_error_rate=0.1)
    with pytest.raises(ValueError, match="same length"):
        apply_thresholds(thresholds, [0.5], [True, False])
