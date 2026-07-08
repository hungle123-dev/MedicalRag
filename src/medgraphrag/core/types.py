from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    options: dict[str, str]
    answer: str


@dataclass(frozen=True)
class RetrievedItem:
    content: str
    score: float
    source: str


@dataclass(frozen=True)
class Prediction:
    qid: str
    choice: str
    evidence: tuple[RetrievedItem, ...]
