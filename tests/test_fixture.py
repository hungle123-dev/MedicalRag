from medgraphrag.data.fixture_dataset import QUESTIONS, CORPUS, TRIPLES
from medgraphrag.core.types import Question


def test_six_questions_all_valid_mcqa():
    assert len(QUESTIONS) == 6
    for q in QUESTIONS:
        assert isinstance(q, Question)
        assert q.answer in q.options


def test_every_question_has_gold_terms():
    for q in QUESTIONS:
        assert q.gold_terms, f"{q.qid} missing gold_terms"


def test_gold_term_is_supported_in_corpus():
    joined = " ".join(CORPUS.values()).lower()
    for q in QUESTIONS:
        for term in q.gold_terms:
            assert term.lower() in joined, f"{q.qid} gold {term!r} not in corpus"


def test_gold_term_is_the_answer_option_text():
    # gold_terms must actually identify the CORRECT option, not a distractor.
    for q in QUESTIONS:
        assert q.options[q.answer].lower() in [t.lower() for t in q.gold_terms]


def test_questions_paraphrase_not_copy_corpus():
    # Guard against fixture rigging: the question text should not contain the
    # gold term verbatim (that would make retrieval trivial).
    for q in QUESTIONS:
        for term in q.gold_terms:
            assert term.lower() not in q.text.lower(), (
                f"{q.qid} leaks gold term {term!r} into the question"
            )


def test_triples_are_triples_with_relation():
    for t in TRIPLES:
        assert len(t) == 3
        assert all(part.strip() for part in t)
