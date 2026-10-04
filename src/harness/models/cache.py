"""On-disk prediction cache, keyed by (task, model, example_id).

Append-only JSONL so a crashed or interrupted run never loses completed
predictions. On load, later rows for the same key win, so re-running a
failed example just appends a corrected row instead of needing a rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Key = tuple[str, str, str]


class PredictionCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[Key, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._entries[_key(record)] = record

    def get(self, task: str, model: str, example_id: str) -> dict[str, Any] | None:
        return self._entries.get((task, model, example_id))

    def put(self, task: str, model: str, example_id: str, record: dict[str, Any]) -> None:
        full_record = {"task": task, "model": model, "example_id": example_id, **record}
        self._entries[(task, model, example_id)] = full_record
        with self.path.open("a") as f:
            f.write(json.dumps(full_record) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


def _key(record: dict[str, Any]) -> Key:
    return (record["task"], record["model"], record["example_id"])
