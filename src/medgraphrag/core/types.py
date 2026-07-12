from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    options: dict[str, str]      # {"A": "...", ...}
    answer: str                  # gold option key
    gold_terms: tuple[str, ...] = ()   # terms identifying correct evidence


@dataclass(frozen=True)
class RetrievedItem:
    content: str
    score: float
    source: str                  # "corpus:<id>" | "triple:<h|r|t>" | "graph:<node>"


@dataclass(frozen=True)
class GraphTriple:
    head: str
    relation: str
    tail: str

    def as_text(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"


@dataclass(frozen=True)
class Prediction:
    qid: str
    choice: str
    evidence: tuple[RetrievedItem, ...] = ()
