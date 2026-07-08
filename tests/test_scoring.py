import pytest

from medgraphrag.eval.accuracy import accuracy
from medgraphrag.core.types import Question, Prediction


def _q(qid, ans):
    return Question(qid, "?", {"A": "a", "B": "b"}, ans)


def _p(qid, choice):
    return Prediction(qid, choice, ())


def test_all_correct_is_one():
    qs = [_q("q1", "A"), _q("q2", "B")]
    assert accuracy([_p("q1", "A"), _p("q2", "B")], qs) == 1.0


def test_half_correct():
    qs = [_q("q1", "A"), _q("q2", "B")]
    assert accuracy([_p("q1", "A"), _p("q2", "A")], qs) == 0.5


def test_mismatched_qid_raises():
    with pytest.raises(ValueError):
        accuracy([_p("qX", "A")], [_q("q1", "A")])
