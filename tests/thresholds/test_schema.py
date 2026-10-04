import jsonschema
import pytest

from harness.thresholds.schema import validate_thresholds_export

VALID_RECORD = {
    "task": "financial_sentiment",
    "model": "llm-mock",
    "tau_low": 0.6,
    "tau_high": 0.9,
    "target": {"coverage": 0.8, "error_rate": 0.1},
    "date": "2026-10-04",
    "run_id": "20261004T000000Z",
}


def test_valid_export_passes() -> None:
    validate_thresholds_export([VALID_RECORD])


def test_missing_required_field_fails() -> None:
    bad = {k: v for k, v in VALID_RECORD.items() if k != "tau_high"}
    with pytest.raises(jsonschema.ValidationError):
        validate_thresholds_export([bad])


def test_tau_out_of_range_fails() -> None:
    bad = {**VALID_RECORD, "tau_low": 1.5}
    with pytest.raises(jsonschema.ValidationError):
        validate_thresholds_export([bad])


def test_unexpected_field_fails() -> None:
    bad = {**VALID_RECORD, "unexpected": "field"}
    with pytest.raises(jsonschema.ValidationError):
        validate_thresholds_export([bad])


def test_not_a_list_fails() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_thresholds_export(VALID_RECORD)  # a dict, not a list
