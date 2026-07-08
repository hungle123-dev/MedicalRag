from medgraphrag.eval.retrieval import recall_at_k, mrr
from medgraphrag.core.types import Question, Prediction, RetrievedItem


def _q(qid, gold):
    return Question(qid, "?", {"A": "a", "B": "b"}, "A", gold_terms=gold)


def _item(content, rank_score=1.0):
    return RetrievedItem(content=content, score=rank_score, source="x")


def test_recall_counts_question_with_gold_term_hit():
    qs = [_q("q1", ("insulin",))]
    ps = [Prediction("q1", "A", (_item("insulin lowers glucose"),))]
    assert recall_at_k(ps, qs) == 1.0


def test_recall_zero_when_gold_absent():
    qs = [_q("q1", ("insulin",))]
    ps = [Prediction("q1", "A", (_item("glucagon raises glucose"),))]
    assert recall_at_k(ps, qs) == 0.0


def test_questions_without_gold_terms_are_ignored():
    qs = [_q("q1", ())]  # no gold -> not scored
    ps = [Prediction("q1", "A", ())]
    assert recall_at_k(ps, qs) == 0.0  # empty scored set


def test_mrr_rewards_higher_rank_of_gold_evidence():
    qs = [_q("q1", ("insulin",))]
    top = [Prediction("q1", "A", (_item("insulin ..."), _item("noise")))]
    second = [Prediction("q1", "A", (_item("noise"), _item("insulin ...")))]
    assert mrr(top, qs) == 1.0
    assert mrr(second, qs) == 0.5
