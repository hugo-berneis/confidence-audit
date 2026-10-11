"""Real Jev client, mirrored from Project 1's `report_qa/decision_jev.py`
(same TypeSafe SDK shapes -- not guessed; see decision.py's module docstring
for why this is a mirror rather than a shared import).

`ask_many()` maps directly onto `system_one()`, Jev's real primitive: several
named questions about one state, answered in a single request. `choice()`,
`score()`, and `noul()` are implemented as single-question calls to
`ask_many()` rather than the other way around, per TypeSafe's own benchmark
cited in Project 1 (~10x faster and ~12x cheaper than asking separately).
"""

from __future__ import annotations

from collections.abc import Mapping

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer as SdkScoreAnswer

from harness.models.decision import ChoiceQuestion, Decision, NoulQuestion, Question, ScoreQuestion

# Mirrors Project 1's knobs -- its own retry/timeout tuning was found live
# against a real, large batch of calls; no reason to re-derive it here.
_TIMEOUT_SECONDS = 30.0
_RETRY_POLICY = RetryPolicy(max_retries=4)

_SINGLE_QUESTION_KEY = "q"


def _to_sdk_question(question: Question) -> Choice | Score | Noul:
    if isinstance(question, ChoiceQuestion):
        return Choice(instructions=question.question, criteria=dict.fromkeys(question.options))
    if isinstance(question, ScoreQuestion):
        return Score(instructions=question.question, criteria=question.criteria)
    if isinstance(question, NoulQuestion):
        return Noul(instructions=question.question)
    raise TypeError(f"Unknown question type: {type(question)!r}")


def _to_decision(
    answer: SdkChoiceAnswer | SdkScoreAnswer | SdkNoulAnswer, question: Question
) -> Decision:
    if isinstance(question, ChoiceQuestion):
        return Decision(value=answer.choice, confidence=answer.confidence)
    if isinstance(question, ScoreQuestion):
        # Score answers land on a 0..len(criteria)-1 rubric position;
        # normalize to the 0-1 scale the rest of the harness expects.
        n = len(question.criteria)
        normalized = answer.score / (n - 1) if n > 1 else answer.score
        return Decision(value=normalized, confidence=answer.confidence)
    if isinstance(question, NoulQuestion):
        # Noul returns one probability, not a (value, confidence) pair: 0.5
        # is "no idea", 0 or 1 is maximally confident either way. Distance
        # from 0.5, rescaled to 0-1, is the natural confidence here.
        probability = answer.noul
        return Decision(value=probability >= 0.5, confidence=abs(probability - 0.5) * 2)
    raise TypeError(f"Unknown question type: {type(question)!r}")


class JevDecisionClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=_TIMEOUT_SECONDS,
            retry=_RETRY_POLICY,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JevDecisionClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask_many(self, state: str, questions: Mapping[str, Question]) -> dict[str, Decision]:
        response = self._client.system_one(
            state=state,
            questions={key: _to_sdk_question(q) for key, q in questions.items()},
        )
        return {key: _to_decision(response.answers[key], q) for key, q in questions.items()}

    def choice(self, state: str, question: str, options: list[str]) -> Decision:
        q = ChoiceQuestion(question=question, options=options)
        return self.ask_many(state, {_SINGLE_QUESTION_KEY: q})[_SINGLE_QUESTION_KEY]

    def score(self, state: str, question: str, criteria: list[str]) -> Decision:
        q = ScoreQuestion(question=question, criteria=criteria)
        return self.ask_many(state, {_SINGLE_QUESTION_KEY: q})[_SINGLE_QUESTION_KEY]

    def noul(self, state: str, question: str) -> Decision:
        q = NoulQuestion(question=question)
        return self.ask_many(state, {_SINGLE_QUESTION_KEY: q})[_SINGLE_QUESTION_KEY]
