from typing import Protocol

from medgraphrag.core.types import Question, Prediction, RetrievedItem, GraphTriple


class LLMClient(Protocol):
    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        ...


class GraphStore(Protocol):
    """Holds the KG. Fixture (test) or LM-extracted-from-MedCorp (real) — the
    pipeline never sees the difference."""

    def triples(self) -> list[GraphTriple]:
        ...

    def neighbors(self, node: str) -> list[GraphTriple]:
        ...


class EntityNormalizer(Protocol):
    """Maps a surface entity to a canonical id. Dict (test) or scispaCy+UMLS
    (real)."""

    def normalize(self, entity: str) -> str:
        ...


class GraphRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        ...


class Fusion(Protocol):
    def fuse(self, ranked_lists: list[list[RetrievedItem]], k: int) -> list[RetrievedItem]:
        ...


class Reranker(Protocol):
    def rerank(self, query: str, items: list[RetrievedItem], k: int) -> list[RetrievedItem]:
        ...
