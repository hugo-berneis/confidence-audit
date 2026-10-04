"""Validates a thresholds export against `configs/thresholds.schema.json`.

CLAUDE.md's Phase 4 "done when" is explicit: the exported file must
validate against its schema, not just happen to have the right fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_PATH = Path("configs/thresholds.schema.json")


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_thresholds_export(
    records: list[dict[str, Any]], schema_path: Path = SCHEMA_PATH
) -> None:
    schema = load_schema(schema_path)
    jsonschema.validate(instance=records, schema=schema)
