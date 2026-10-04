"""Loader for the 10-K risk-topic classification task.

Project 1's `exports/paragraph_labels.jsonl` is Jev's own tagger output
(every row has `topic_confidence == 0.6`, the value `tagger.py` got back
from the real Jev client) -- CLAUDE.md is explicit that Jev's own tags can
never be gold for evaluating Jev. The gold set here is instead a small
human-labeled sample; see `scripts/prepare_10k_gold_sample.py` for how it's
drawn from Project 1's raw paragraphs, and `data/gold/README.md` for how to
label it.

`group` is the ticker, matching the confirmed OOD definition: the OOD split
holds out whole companies.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.data.schema import Example

TASK = "risk_topic"

# Fixed label set, mirrored from Project 1's config/topics.yaml -- Jev is
# never allowed to answer outside this list, so gold labels must use the
# same set or accuracy comparisons would be meaningless.
TOPICS = [
    "liquidity",
    "credit",
    "regulatory",
    "cyber",
    "macro",
    "competition",
    "operations",
    "legal",
    "supply_chain",
    "labor",
    "other",
]


def load(gold_path: str | Path) -> list[Example]:
    """Load the hand-labeled gold sample. Raises if any row is still unlabeled."""
    path = Path(gold_path)
    examples = []
    unlabeled = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if not row["label"]:
                unlabeled.append(row["paragraph_id"])
                continue
            if row["label"] not in TOPICS:
                raise ValueError(f"{row['paragraph_id']}: label {row['label']!r} not in TOPICS")
            examples.append(
                Example(
                    id=row["paragraph_id"],
                    text=row["text"],
                    label=row["label"],
                    group=row["ticker"],
                )
            )
    if unlabeled:
        raise ValueError(
            f"{len(unlabeled)} rows in {path} still have no label: {unlabeled[:5]}..."
        )
    return examples
