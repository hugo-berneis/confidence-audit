from harness.data.schema import Example
from harness.data.splits import split_by_group, split_random


def _examples(n: int, label: str, group: str) -> list[Example]:
    return [
        Example(id=f"{group}-{label}-{i}", text="text", label=label, group=group)
        for i in range(n)
    ]


def test_split_by_group_holds_out_whole_groups() -> None:
    examples = (
        _examples(10, "a", "company-1")
        + _examples(10, "b", "company-1")
        + _examples(10, "a", "company-2")
        + _examples(10, "b", "company-2")
        + _examples(5, "a", "company-3")
        + _examples(5, "b", "company-3")
    )
    result = split_by_group(examples, ood_groups={"company-3"}, seed=0)

    assert {e.group for e in result.ood} == {"company-3"}
    assert "company-3" not in {e.group for e in result.tune}
    assert "company-3" not in {e.group for e in result.test}
    assert len(result.ood) == 10


def test_split_by_group_no_id_appears_twice() -> None:
    examples = _examples(20, "a", "company-1") + _examples(20, "a", "company-2")
    result = split_by_group(examples, ood_groups={"company-2"}, seed=0)

    tune_ids = {e.id for e in result.tune}
    test_ids = {e.id for e in result.test}
    ood_ids = {e.id for e in result.ood}
    assert not (tune_ids & test_ids)
    assert not (tune_ids & ood_ids)
    assert not (test_ids & ood_ids)
    assert tune_ids | test_ids | ood_ids == {e.id for e in examples}


def test_split_by_group_preserves_label_balance_in_tune_and_test() -> None:
    examples = _examples(100, "a", "company-1") + _examples(50, "b", "company-1")
    result = split_by_group(examples, ood_groups=set(), tune_fraction=0.7, seed=0)

    tune_labels = [e.label for e in result.tune]
    test_labels = [e.label for e in result.test]
    assert tune_labels.count("a") == 70
    assert tune_labels.count("b") == 35
    assert test_labels.count("a") == 30
    assert test_labels.count("b") == 15


def test_split_random_has_no_ood() -> None:
    examples = _examples(10, "a", "g1") + _examples(10, "b", "g2")
    result = split_random(examples, seed=0)
    assert result.ood == []
    assert len(result.tune) + len(result.test) == len(examples)


def test_split_random_no_id_appears_twice() -> None:
    examples = _examples(50, "a", "g1")
    result = split_random(examples, seed=0)
    tune_ids = {e.id for e in result.tune}
    test_ids = {e.id for e in result.test}
    assert not (tune_ids & test_ids)
    assert tune_ids | test_ids == {e.id for e in examples}


def test_split_is_deterministic_given_seed() -> None:
    examples = _examples(30, "a", "g1") + _examples(30, "b", "g2")
    first = split_random(examples, seed=42)
    second = split_random(examples, seed=42)
    assert [e.id for e in first.tune] == [e.id for e in second.tune]
    assert [e.id for e in first.test] == [e.id for e in second.test]
