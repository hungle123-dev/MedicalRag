from medgraphrag.retrieval.null import NullRetriever
from medgraphrag.retrieval.text import TextRetriever
from medgraphrag.retrieval.triple import TripleRetriever
from tests.fixtures.mini_mirage import CORPUS, TRIPLES


def test_null_returns_nothing():
    assert NullRetriever().retrieve("anything", k=5) == []


def test_text_retriever_ranks_relevant_doc_first():
    items = TextRetriever(CORPUS).retrieve("antibiotic for strep throat", k=1)
    assert len(items) == 1
    assert "penicillin" in items[0].content.lower()
    assert items[0].source.startswith("corpus:")


def test_triple_retriever_matches_on_overlap():
    items = TripleRetriever(TRIPLES).retrieve("what lowers blood glucose", k=1)
    assert len(items) == 1
    assert "insulin" in items[0].content.lower()
    assert items[0].source.startswith("triple:")
