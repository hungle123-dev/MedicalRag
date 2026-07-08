from medgraphrag.core.types import RetrievedItem


class NullRetriever:
    """E0: no retrieval."""

    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        return []
