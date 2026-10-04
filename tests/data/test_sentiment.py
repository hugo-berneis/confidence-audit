from harness.data.sentiment import _to_examples

RAW_LINES = [
    "Profits fell sharply this quarter.@negative",
    "The factory will open on schedule.@neutral",
    "Revenue beat expectations significantly.@positive",
]


def test_to_examples_preserves_labels() -> None:
    examples = _to_examples(RAW_LINES)
    assert [e.label for e in examples] == ["negative", "neutral", "positive"]


def test_to_examples_ids_are_unique_and_group_equals_id() -> None:
    examples = _to_examples(RAW_LINES)
    ids = [e.id for e in examples]
    assert len(set(ids)) == len(ids)
    assert all(e.group == e.id for e in examples)


def test_to_examples_preserves_text() -> None:
    examples = _to_examples(RAW_LINES)
    assert examples[0].text == "Profits fell sharply this quarter."


def test_to_examples_handles_at_signs_in_sentence_text() -> None:
    examples = _to_examples(["Reach us @ info@example.com for details.@neutral"])
    assert examples[0].text == "Reach us @ info@example.com for details."
    assert examples[0].label == "neutral"
