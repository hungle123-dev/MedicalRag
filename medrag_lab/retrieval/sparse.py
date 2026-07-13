from __future__ import annotations

from medrag_lab.indexing.bm25 import BM25Index
from medrag_lab.schemas import RetrievedDocument


class SparseRetriever:
    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, k: int = 100) -> tuple[list[RetrievedDocument], float]:
        return self.index.search(query, k)
