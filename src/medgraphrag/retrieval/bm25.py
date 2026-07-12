"""E1 retriever. Backed by bm25s (C-accelerated) — query is <10ms over 125k
chunks vs ~5s for pure-python rank_bm25, so E1 runs at E0 speed and no
retrieval cache is needed. Same Retriever protocol as before."""
import bm25s

from medgraphrag.core.types import RetrievedItem
from medgraphrag.retrieval.tokenize import tokenize


class BM25Retriever:
    def __init__(self, corpus: dict[str, str]):
        self._ids = list(corpus.keys())
        self._docs = [corpus[i] for i in self._ids]
        if self._docs:
            tokenized = [tokenize(d) for d in self._docs]
            self._bm25 = bm25s.BM25()
            self._bm25.index(tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if self._bm25 is None or k <= 0 or not query.strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        k = min(k, len(self._docs))
        idxs, scores = self._bm25.retrieve([q_tokens], k=k, show_progress=False)
        out = []
        for rank in range(idxs.shape[1]):
            i = int(idxs[0, rank])
            score = float(scores[0, rank])
            if score <= 0.0:
                break
            out.append(RetrievedItem(content=self._docs[i], score=score,
                                     source=f"corpus:{self._ids[i]}"))
        return out
