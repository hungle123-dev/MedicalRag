from medgraphrag.core.types import Question, RetrievedItem, Prediction


def test_question_holds_mcqa_fields():
    q = Question(qid="q1", text="What?", options={"A": "x", "B": "y"}, answer="B")
    assert q.answer == "B"
    assert q.options["A"] == "x"


def test_prediction_carries_evidence():
    item = RetrievedItem(content="fact", score=1.0, source="corpus:d1")
    p = Prediction(qid="q1", choice="B", evidence=(item,))
    assert p.choice == "B"
    assert p.evidence[0].source == "corpus:d1"
