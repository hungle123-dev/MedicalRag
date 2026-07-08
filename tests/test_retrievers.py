from medgraphrag.retrieval.null import NullRetriever
from medgraphrag.retrieval.text import TextRetriever
from medgraphrag.retrieval.triple import TripleRetriever
from medgraphrag.data.fixture_dataset import CORPUS, TRIPLES


def test_null_returns_nothing():
    assert NullRetriever().retrieve("anything", k=5) == []


def test_text_retriever_finds_gold_doc_in_topk():
    items = TextRetriever(CORPUS).retrieve("agent for streptococcal pharyngitis", k=3)
    assert any("penicillin" in i.content.lower() for i in items)
    assert all(i.source.startswith("corpus:") for i in items)


def test_text_retriever_empty_corpus_returns_empty():
    assert TextRetriever({}).retrieve("anything", k=3) == []


def test_text_retriever_empty_query_returns_empty():
    assert TextRetriever(CORPUS).retrieve("   ", k=3) == []


def test_text_retriever_k_larger_than_corpus_is_capped():
    items = TextRetriever(CORPUS).retrieve("penicillin insulin warfarin aspirin", k=999)
    assert len(items) <= len(CORPUS)


def test_text_retriever_never_returns_zero_score_docs():
    items = TextRetriever(CORPUS).retrieve("penicillin", k=len(CORPUS))
    assert all(i.score > 0.0 for i in items)


def test_triple_retriever_uses_relation_to_disambiguate_shared_head():
    r = TripleRetriever(TRIPLES)
    reduced = r.retrieve("which hormone reduces serum glucose", k=1)
    elevated = r.retrieve("which hormone elevates serum glucose", k=1)
    assert "insulin" in reduced[0].content.lower()
    assert "glucagon" in elevated[0].content.lower()


def test_triple_retriever_empty_query_returns_empty():
    assert TripleRetriever(TRIPLES).retrieve("", k=3) == []
