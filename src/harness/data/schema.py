"""Common schema every task loader produces, so splits/models/metrics code
never needs to know which task it's looking at.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    group: str
    """Unit the OOD split holds out wholesale (e.g. a ticker). For tasks with
    no real grouping, this is just `id`, so every example is its own group
    and `split_by_group` has nothing meaningful to hold out -- use
    `split_random` for those instead."""


@dataclass(frozen=True)
class SplitResult:
    tune: list[Example]
    test: list[Example]
    ood: list[Example]
