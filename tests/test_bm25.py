import json

from medrag_lab.indexing.bm25 import BM25Index


def test_bm25_build_search_and_local_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"id": "1", "title": "Asthma treatment", "text": "Bronchodilator therapy", "url": "u1"},
        {"id": "2", "title": "Diabetes", "text": "Insulin therapy", "url": "u2"},
        {"id": "3", "title": "Hypertension", "text": "Blood pressure therapy", "url": "u3"},
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index = BM25Index.build(corpus)
    path = tmp_path / "index.pkl"
    index.save(path)
    results, latency = BM25Index.load(path).search("asthma bronchodilator", 1)
    assert results[0].pmid == "1"
    assert latency >= 0
