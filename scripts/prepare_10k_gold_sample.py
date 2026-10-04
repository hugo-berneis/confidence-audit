#!/usr/bin/env python3
"""Draws a stratified sample of Project 1's raw 10-K paragraphs for Hugo to
hand-label, and writes it to `risk_topic_gold_path` with `label` left blank.

Project 1's own tags (`exports/paragraph_labels.jsonl`) can't be used as gold
-- see `harness/data/risk_topics.py`. This script only reads Project 1's raw,
untagged paragraphs (`data/processed/paragraphs.jsonl`).

Usage:
    uv run scripts/prepare_10k_gold_sample.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from harness.config import get_settings

PER_STRATUM = 10  # ticker x section stratum -> ~120 rows total at this MVP's scale


def main() -> None:
    settings = get_settings()
    paragraphs_path = Path(settings.risk_topic_paragraphs_path)
    out_path = Path(settings.risk_topic_gold_path)

    if out_path.exists():
        raise SystemExit(f"{out_path} already exists -- delete it first if you want a new sample")

    rows = [json.loads(line) for line in paragraphs_path.open()]

    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_stratum[(row["ticker"], row["section"])].append(row)

    rng = random.Random(settings.seed)
    sample = []
    for key in sorted(by_stratum):
        stratum_rows = by_stratum[key].copy()
        rng.shuffle(stratum_rows)
        sample.extend(stratum_rows[:PER_STRATUM])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in sample:
            f.write(
                json.dumps(
                    {
                        "paragraph_id": row["paragraph_id"],
                        "ticker": row["ticker"],
                        "fiscal_year": row["fiscal_year"],
                        "section": row["section"],
                        "text": row["text"],
                        "label": "",
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(sample)} rows to {out_path} for hand-labeling.")
    print("See data/gold/README.md for the label set and instructions.")


if __name__ == "__main__":
    main()
