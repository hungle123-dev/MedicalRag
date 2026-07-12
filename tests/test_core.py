import pytest

from medgraphrag.core.types import Question, RetrievedItem, GraphTriple, Prediction
from medgraphrag.core import registry


def test_question_mcqa_shape():
    q = Question("q1", "What?", {"A": "x", "B": "y"}, "B", gold_terms=("y",))
    assert q.answer == "B"
    assert q.gold_terms == ("y",)


def test_triple_as_text():
    t = GraphTriple("aspirin", "inhibits", "COX")
    assert t.as_text() == "aspirin inhibits COX"


def test_prediction_defaults_empty_evidence():
    p = Prediction("q1", "A")
    assert p.evidence == ()


def test_registry_build_and_reject_unknown():
    @registry.retriever("dummy")
    def _build(ctx):
        return ("built", ctx["x"])

    assert registry.build("retriever", "dummy", {"x": 1}) == ("built", 1)
    assert "dummy" in registry.names("retriever")
    with pytest.raises(ValueError):
        registry.build("retriever", "nope", {})


def test_registry_rejects_duplicate():
    @registry.fusion("dup")
    def _a(ctx):
        return 1
    with pytest.raises(ValueError):
        @registry.fusion("dup")
        def _b(ctx):
            return 2
