from medgraphrag.core.types import RetrievedItem
from medgraphrag.retrieval.tokenize import tokenize


class TripleRetriever:
    """E2: overlap of query tokens against head+relation+tail of each triple.

    Relation IS scored (not just head+tail) so triples sharing a head — e.g.
    'serum glucose reduced_by insulin' vs 'serum glucose elevated_by glucagon'
    — are disambiguated by the relation wording in the question. Deterministic
    tie-break by original triple order keeps runs reproducible.
    """

    def __init__(self, triples: list[tuple[str, str, str]]):
        self._triples = triples

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        if k <= 0 or not query.strip():
            return []
        q = set(tokenize(query))
        scored = []
        for idx, (h, r, t) in enumerate(self._triples):
            terms = set(tokenize(h)) | set(tokenize(r)) | set(tokenize(t))
            overlap = len(q & terms)
            if overlap:
                # sort key: higher overlap first, then original order (stable)
                scored.append((-overlap, idx, h, r, t))
        scored.sort()
        out = []
        for neg_overlap, idx, h, r, t in scored[:k]:
            out.append(RetrievedItem(content=f"{h} {r} {t}", score=float(-neg_overlap),
                                     source=f"triple:{h}|{r}|{t}"))
        return out
