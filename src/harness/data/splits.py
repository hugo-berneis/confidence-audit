"""Tune/test/OOD splitting.

Both split functions stratify by label within whatever pool they're splitting,
so label balance holds across tune and test -- see `scripts/describe_splits.py`
for the table this is checked with.
"""

from __future__ import annotations

import random
from collections import defaultdict

from harness.data.schema import Example, SplitResult


def split_by_group(
    examples: list[Example],
    ood_groups: set[str],
    tune_fraction: float = 0.7,
    seed: int = 0,
) -> SplitResult:
    """Hold out every example whose `group` is in `ood_groups` as OOD; split
    the remaining examples into tune/test, stratified by label.
    """
    ood = [e for e in examples if e.group in ood_groups]
    in_distribution = [e for e in examples if e.group not in ood_groups]
    tune, test = _stratified_split(in_distribution, tune_fraction, seed)
    return SplitResult(tune=tune, test=test, ood=ood)


def split_random(
    examples: list[Example],
    tune_fraction: float = 0.7,
    seed: int = 0,
) -> SplitResult:
    """Stratified tune/test split with no OOD split (`ood` is always empty).

    For tasks with no real grouping to hold out -- see `Example.group`.
    """
    tune, test = _stratified_split(examples, tune_fraction, seed)
    return SplitResult(tune=tune, test=test, ood=[])


def _stratified_split(
    examples: list[Example], tune_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    rng = random.Random(seed)
    tune: list[Example] = []
    test: list[Example] = []
    for label in sorted(by_label):
        group = by_label[label].copy()
        rng.shuffle(group)
        cut = round(len(group) * tune_fraction)
        tune.extend(group[:cut])
        test.extend(group[cut:])
    return tune, test
