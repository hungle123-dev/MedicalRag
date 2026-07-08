from rank_bm25 import BM25Okapi

from medgraphrag.core.types import RetrievedItem
from medgraphrag.retrieval.tokenize import tokenize


class TextRetriever:
    """E1: BM25 over a flat document corpus."""

    def __init__(self, corpus: dict[str, str]):
        self._ids = list(corpus.keys())
        self._docs = [corpus[i] for i in self._ids]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._docs]) if self._docs else None

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if self._bm25 is None or k <= 0 or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        out = []
        for i in ranked[:k]:
            if scores[i] <= 0.0:
                break  # don't return zero-relevance docs as if they were hits
            out.append(RetrievedItem(content=self._docs[i], score=float(scores[i]),
                                     source=f"corpus:{self._ids[i]}"))
        return out
