from typing import Protocol

from medgraphrag.core.types import Question, Prediction, RetrievedItem


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedItem]:
        ...


class LLMClient(Protocol):
    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        ...


class Arm(Protocol):
    def answer(self, q: Question) -> Prediction:
        ...
