from harness.models.decision import DecisionClient, MockDecisionClient


def test_mock_implements_decision_client_protocol() -> None:
    assert isinstance(MockDecisionClient(), DecisionClient)


def test_choice_is_deterministic() -> None:
    client = MockDecisionClient()
    first = client.choice("some 10-K paragraph", "What risk topic is this?", ["A", "B", "C"])
    second = client.choice("some 10-K paragraph", "What risk topic is this?", ["A", "B", "C"])
    assert first == second
    assert first.value in {"A", "B", "C"}


def test_choice_rejects_empty_options() -> None:
    client = MockDecisionClient()
    try:
        client.choice("state", "question", [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty options")


def test_different_states_can_choose_different_options() -> None:
    client = MockDecisionClient()
    options = ["A", "B", "C", "D", "E"]
    values = {client.choice(f"state-{i}", "question", options).value for i in range(20)}
    assert len(values) > 1


def test_score_returns_value_in_unit_interval() -> None:
    client = MockDecisionClient()
    decision = client.score("state", "question", ["low", "medium", "high"])
    assert 0.0 <= decision.value < 1.0


def test_noul_returns_bool() -> None:
    client = MockDecisionClient()
    decision = client.noul("state", "question")
    assert isinstance(decision.value, bool)


def test_confidence_is_configurable() -> None:
    client = MockDecisionClient(confidence=0.9)
    decision = client.noul("state", "question")
    assert decision.confidence == 0.9
