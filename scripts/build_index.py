"""Builds the BM25 index over the FULL MedRAG textbook corpus (125k chunks) and
pickles it so E0/E1 x 2-model runs reuse it instead of re-tokenizing each time.

ponytail: pickle cache; delete data/bm25_index.pkl to force rebuild.
Run: python scripts/build_index.py
"""
import pickle
import time
from pathlib import Path

from medgraphrag.data.corpus_loader import load_textbook_corpus
from medgraphrag.retrieval.bm25 import BM25Retriever

CORPUS_DIR = "data/raw/medrag_textbooks/chunk"
OUT = "data/bm25_index.pkl"


def build() -> BM25Retriever:
    t0 = time.time()
    corpus = load_textbook_corpus(CORPUS_DIR)  # full corpus, no limit
    print(f"loaded {len(corpus)} chunks in {time.time()-t0:.1f}s")
    t1 = time.time()
    retriever = BM25Retriever(corpus)
    print(f"built BM25 index in {time.time()-t1:.1f}s")
    return retriever


def main():
    retriever = build()
    t0 = time.time()
    with open(OUT, "wb") as f:
        pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = Path(OUT).stat().st_size / 1e6
    print(f"pickled -> {OUT} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")

    # sanity: reload + one query
    with open(OUT, "rb") as f:
        r2 = pickle.load(f)
    hits = r2.retrieve("aspirin cyclooxygenase inhibition platelet", k=3)
    print(f"reload OK, sample query returned {len(hits)} hits; top source: "
          f"{hits[0].source if hits else 'NONE'}")


if __name__ == "__main__":
    main()
