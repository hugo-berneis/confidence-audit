import json
from pathlib import Path

import pytest

from harness.data.risk_topics import TOPICS, load

ROWS = [
    {
        "paragraph_id": "AAPL-2024-risk_factors-0",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "section": "risk_factors",
        "text": "We depend on single suppliers for some components.",
        "label": "supply_chain",
    },
    {
        "paragraph_id": "WMT-2024-mdna-3",
        "ticker": "WMT",
        "fiscal_year": 2024,
        "section": "mdna",
        "text": "New regulations could increase our compliance costs.",
        "label": "regulatory",
    },
]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_maps_to_common_schema(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    _write(path, ROWS)

    examples = load(path)

    assert len(examples) == 2
    first = examples[0]
    assert first.id == "AAPL-2024-risk_factors-0"
    assert first.label == "supply_chain"
    assert first.group == "AAPL"


def test_load_rejects_blank_labels(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    _write(path, [*ROWS, {**ROWS[0], "paragraph_id": "AAPL-2024-risk_factors-1", "label": ""}])

    with pytest.raises(ValueError, match="still have no label"):
        load(path)


def test_load_rejects_labels_outside_topic_set(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    _write(path, [{**ROWS[0], "label": "not_a_real_topic"}])

    with pytest.raises(ValueError, match="not in TOPICS"):
        load(path)


def test_topics_list_has_eleven_entries() -> None:
    assert len(TOPICS) == 11
    assert len(set(TOPICS)) == 11
