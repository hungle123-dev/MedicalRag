"""BM25 tested against the real MedRAG textbook corpus (a small slice), not a
synthetic fixture — we want to know it actually finds relevant chunks."""
from pathlib import Path

import pytest

from medgraphrag.retrieval.bm25 import BM25Retriever
from medgraphrag.data.corpus_loader import load_textbook_corpus

TEXTBOOKS_DIR = "data/raw/medrag_textbooks/chunk"

pytestmark = pytest.mark.skipif(
    not Path(TEXTBOOKS_DIR).exists(),
    reason="real MedRAG textbook corpus not downloaded",
)


def test_empty_corpus_returns_empty():
    assert BM25Retriever({}).retrieve("anything", k=3) == []


def test_empty_query_returns_empty():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, books=["Pharmacology_Katzung"], limit=200)
    assert BM25Retriever(corpus).retrieve("   ", k=3) == []


def test_finds_relevant_chunk_in_real_pharmacology_corpus():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, books=["Pharmacology_Katzung"], limit=2000)
    r = BM25Retriever(corpus)
    items = r.retrieve("mechanism of action of aspirin cyclooxygenase inhibition", k=5)
    assert len(items) > 0
    assert all(i.source.startswith("corpus:") for i in items)
    # top hits should be actually about the query topic, not noise
    joined = " ".join(i.content.lower() for i in items)
    assert "aspirin" in joined or "cyclooxygenase" in joined or "cox" in joined


def test_k_larger_than_corpus_is_capped():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, books=["Pathoma_Husain"], limit=50)
    items = BM25Retriever(corpus).retrieve("cancer cell proliferation tumor", k=999)
    assert len(items) <= 50


def test_never_returns_zero_score_docs():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, books=["Pathoma_Husain"], limit=500)
    items = BM25Retriever(corpus).retrieve("insulin glucose diabetes", k=20)
    assert all(i.score > 0.0 for i in items)
