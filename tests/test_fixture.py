from tests.fixtures.mini_mirage import QUESTIONS, CORPUS, TRIPLES
from medgraphrag.core.types import Question


def test_six_questions_all_valid_mcqa():
    assert len(QUESTIONS) == 6
    for q in QUESTIONS:
        assert isinstance(q, Question)
        assert q.answer in q.options


def test_each_answer_supported_somewhere_in_corpus():
    joined = " ".join(CORPUS.values()).lower()
    for q in QUESTIONS:
        assert q.options[q.answer].lower() in joined


def test_triples_are_triples():
    for t in TRIPLES:
        assert len(t) == 3
