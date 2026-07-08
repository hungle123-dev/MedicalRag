from medgraphrag.llm.mock import MockLLM


def test_mock_picks_option_whose_text_is_in_context():
    llm = MockLLM(default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    assert llm.choose("treatment?", opts, context="give penicillin now") == "B"


def test_mock_falls_back_to_default_without_context_hit():
    llm = MockLLM(default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    assert llm.choose("treatment?", opts, context="no useful text") == "A"


def test_mock_rules_override_default():
    llm = MockLLM(rules={"fever": "B"}, default="A")
    opts = {"A": "aspirin", "B": "penicillin"}
    assert llm.choose("patient has fever", opts, context="") == "B"
