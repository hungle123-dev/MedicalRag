import pytest

from medgraphrag.eval.accuracy import accuracy
from medgraphrag.eval.retrieval import retrieval_recall, mrr
from medgraphrag.core.types import Question, Prediction, RetrievedItem


def _q(qid, ans, gold=()):
    return Question(qid, "?", {"A": "a", "B": "b"}, ans, gold_terms=gold)


def _p(qid, choice, evidence=()):
    return Prediction(qid, choice, evidence)


def test_accuracy_all_correct():
    qs = [_q("q1", "A"), _q("q2", "B")]
    assert accuracy([_p("q1", "A"), _p("q2", "B")], qs) == 1.0


def test_accuracy_half_correct():
    qs = [_q("q1", "A"), _q("q2", "B")]
    assert accuracy([_p("q1", "A"), _p("q2", "A")], qs) == 0.5


def test_accuracy_count_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([_p("q1", "A")], [_q("q1", "A"), _q("q2", "B")])


def test_recall_hit_and_miss():
    qs = [_q("q1", "A", gold=("insulin",))]
    hit = [_p("q1", "A", (RetrievedItem("insulin lowers glucose", 1.0, "corpus:d1"),))]
    miss = [_p("q1", "A", (RetrievedItem("glucagon raises glucose", 1.0, "corpus:d2"),))]
    assert retrieval_recall(hit, qs) == 1.0
    assert retrieval_recall(miss, qs) == 0.0


def test_mrr_rewards_higher_rank():
    qs = [_q("q1", "A", gold=("insulin",))]
    top = [_p("q1", "A", (RetrievedItem("insulin ...", 1.0, "x"), RetrievedItem("noise", 1.0, "x")))]
    second = [_p("q1", "A", (RetrievedItem("noise", 1.0, "x"), RetrievedItem("insulin ...", 1.0, "x")))]
    assert mrr(top, qs) == 1.0
    assert mrr(second, qs) == 0.5


def test_accuracy_and_recall_are_independent_signals():
    # retrieval finds the gold term (recall hit) but the answer is still wrong
    qs = [_q("q1", "B", gold=("insulin",))]
    preds = [_p("q1", "A", (RetrievedItem("insulin lowers glucose", 1.0, "corpus:d1"),))]
    assert retrieval_recall(preds, qs) == 1.0
    assert accuracy(preds, qs) == 0.0
