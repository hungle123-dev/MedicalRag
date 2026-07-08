from medgraphrag.core.types import RetrievedItem
from medgraphrag.retrieval.tokenize import tokenize


class TripleRetriever:
    """E2: token-overlap match of query against (head, tail) of KG triples."""

    def __init__(self, triples: list[tuple[str, str, str]]):
        self._triples = triples

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        q = set(tokenize(query))
        scored = []
        for h, r, t in self._triples:
            terms = set(tokenize(h)) | set(tokenize(t))
            overlap = len(q & terms)
            if overlap:
                scored.append((overlap, h, r, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for overlap, h, r, t in scored[:k]:
            out.append(RetrievedItem(content=f"{h} {r} {t}", score=float(overlap),
                                     source=f"triple:{h}|{r}|{t}"))
        return out
