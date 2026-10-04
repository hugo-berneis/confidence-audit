"""`DecisionClient` interface, mirrored from Project 1 (10k-analyst's
`report_qa.decision` / `report_qa.decision_mock`, as of 2026-10-04).

This harness evaluates Jev, so it must speak the exact same interface Project
1 uses -- not a guessed one. Vendored rather than imported as a path
dependency so this repo stays light (Project 1 pulls in fastapi, pgvector,
sentence-transformers, streamlit, psycopg transitively; none of that is
needed here). If Project 1's interface changes, update this file to match
and note it in CLAUDE.md's decisions log -- there is no automatic sync.

The real `JevDecisionClient` is intentionally not vendored: Phase 0 only
needs the mock. Wire in the real client in Phase 2 if Jev's results are
in scope, using the same TypeSafe SDK pattern as Project 1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Decision:
    """A single Jev decision: an answer plus how confident Jev is in it."""

    value: str | float | bool
    confidence: float


@dataclass(frozen=True)
class ChoiceQuestion:
    question: str
    options: list[str]


@dataclass(frozen=True)
class ScoreQuestion:
    question: str
    criteria: list[str]


@dataclass(frozen=True)
class NoulQuestion:
    question: str


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@runtime_checkable
class DecisionClient(Protocol):
    """Everything this harness is allowed to ask Jev to do."""

    def choice(self, state: str, question: str, options: list[str]) -> Decision:
        """Pick one of `options` for `question`, given context `state`."""
        ...

    def score(self, state: str, question: str, criteria: list[str]) -> Decision:
        """Place `state` on the ordered rubric `criteria` (low to high), for `question`."""
        ...

    def noul(self, state: str, question: str) -> Decision:
        """Answer a yes/no question."""
        ...

    def ask_many(self, state: str, questions: Mapping[str, Question]) -> dict[str, Decision]:
        """Answer several named questions about the same `state` in one round trip."""
        ...


def _stable_unit_interval(*parts: str) -> float:
    """Map arbitrary strings to a reproducible float in [0, 1)."""
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockDecisionClient:
    """Deterministic stand-in for Jev. Confidence is fixed and configurable."""

    def __init__(self, confidence: float = 0.6) -> None:
        self.confidence = confidence

    def choice(self, state: str, question: str, options: list[str]) -> Decision:
        if not options:
            raise ValueError("choice() requires at least one option")
        index = int(_stable_unit_interval(state, question, "choice") * len(options))
        return Decision(value=options[min(index, len(options) - 1)], confidence=self.confidence)

    def score(self, state: str, question: str, criteria: list[str]) -> Decision:
        if not criteria:
            raise ValueError("score() requires at least one criterion")
        value = _stable_unit_interval(state, question, "score", str(len(criteria)))
        return Decision(value=value, confidence=self.confidence)

    def noul(self, state: str, question: str) -> Decision:
        value = _stable_unit_interval(state, question, "noul") >= 0.5
        return Decision(value=value, confidence=self.confidence)

    def ask_many(self, state: str, questions: Mapping[str, Question]) -> dict[str, Decision]:
        """No real batching to gain here -- just answers each one locally."""
        answers = {}
        for key, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                answers[key] = self.choice(state, q.question, q.options)
            elif isinstance(q, ScoreQuestion):
                answers[key] = self.score(state, q.question, q.criteria)
            elif isinstance(q, NoulQuestion):
                answers[key] = self.noul(state, q.question)
            else:
                raise TypeError(f"Unknown question type: {type(q)!r}")
        return answers


def get_decision_client(jev_api_key: str) -> DecisionClient:
    """Real Jev if `jev_api_key` is set, otherwise the deterministic mock.

    Real-client wiring isn't implemented yet (Phase 0 scope is the mock) --
    calling this with a key raises until Phase 2 adds it.
    """
    if jev_api_key:
        raise NotImplementedError(
            "Real JevDecisionClient isn't wired in yet. Phase 0 scope is the mock only."
        )
    return MockDecisionClient()
