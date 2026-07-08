import pytest

from medgraphrag.core.registry import build
from medgraphrag.llm.mock import MockLLM
from medgraphrag.data.fixture_dataset import QUESTIONS, CORPUS, TRIPLES

# Ensure E0/E1/E2 are registered.
import medgraphrag.arms.standard  # noqa: F401,E402


def _ctx(k=3):
    return {"corpus": CORPUS, "triples": TRIPLES, "llm": MockLLM(default="A"), "k": k}


def _by_qid(preds):
    return {p.qid: p for p in preds}


def test_e0_never_retrieves_and_answers_default():
    arm = build("E0", _ctx())
    for q in QUESTIONS:
        pred = arm.answer(q)
        assert pred.evidence == ()
        assert pred.choice == "A"  # MockLLM default with no context


def test_e1_retrieves_corpus_evidence_for_every_question():
    arm = build("E1", _ctx())
    for q in QUESTIONS:
        pred = arm.answer(q)
        assert pred.evidence, f"E1 returned no evidence for {q.qid}"
        assert all(e.source.startswith("corpus:") for e in pred.evidence)


def test_e2_top_triple_is_the_gold_triple_for_each_question():
    # Structured triples are unambiguous: the top evidence must contain the gold term.
    arm = build("E2", _ctx())
    for q in QUESTIONS:
        pred = arm.answer(q)
        assert pred.evidence
        top = pred.evidence[0].content.lower()
        assert q.gold_terms[0].lower() in top, f"{q.qid}: {top!r} missing {q.gold_terms}"


def test_e2_disambiguates_shared_head_by_relation():
    # q2 (reduces glucose) and q5 (elevates glucose) share head "serum glucose";
    # only the relation separates insulin from glucagon.
    arm = build("E2", _ctx())
    preds = _by_qid([arm.answer(q) for q in QUESTIONS])
    assert "insulin" in preds["q2"].evidence[0].content.lower()
    assert "glucagon" in preds["q5"].evidence[0].content.lower()


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError):
        build("E9", _ctx())
