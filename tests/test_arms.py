import pytest

import medgraphrag.arms.standard  # noqa: F401  (registers E0/E1/E2)
from medgraphrag.core.registry import build
from medgraphrag.llm.mock import MockLLM
from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES


def _ctx():
    return {"corpus": CORPUS, "triples": TRIPLES, "llm": MockLLM(default="A"), "k": 3}


def test_e0_uses_no_context_falls_to_default():
    pred = build("E0", _ctx()).answer(QUESTIONS[1])  # q2 gold "B"
    assert pred.choice == "A"
    assert pred.evidence == ()


def test_e1_text_retrieval_supplies_context_and_fixes_answer():
    pred = build("E1", _ctx()).answer(QUESTIONS[1])
    assert pred.choice == "B"
    assert any(e.source.startswith("corpus:") for e in pred.evidence)


def test_e2_triple_retrieval_supplies_context():
    pred = build("E2", _ctx()).answer(QUESTIONS[1])
    assert pred.choice == "B"
    assert any(e.source.startswith("triple:") for e in pred.evidence)


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError):
        build("E9", _ctx())
