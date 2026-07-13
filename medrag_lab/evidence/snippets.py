from __future__ import annotations

import re
from dataclasses import dataclass

from medrag_lab.indexing.bm25 import tokenize
from medrag_lab.schemas import RetrievedDocument

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Snippet:
    pmid: str
    title: str
    text: str
    score: float
    url: str


def sentence_windows(document: RetrievedDocument, size: int = 3) -> list[Snippet]:
    sentences = [value.strip() for value in SENTENCE.split(document.text) if value.strip()]
    if not sentences:
        sentences = [document.text]
    return [
        Snippet(
            pmid=document.pmid,
            title=document.title,
            text=" ".join(sentences[start : start + size]),
            score=document.score,
            url=document.url,
        )
        for start in range(len(sentences))
    ]


def rank_snippets(
    question: str, documents: list[RetrievedDocument], limit: int = 20
) -> list[Snippet]:
    query_terms = set(tokenize(question))
    candidates: list[Snippet] = []
    for document in documents:
        for snippet in sentence_windows(document):
            terms = set(tokenize(snippet.text))
            lexical = len(query_terms & terms) / max(len(query_terms), 1)
            candidates.append(
                Snippet(
                    pmid=snippet.pmid,
                    title=snippet.title,
                    text=snippet.text,
                    score=lexical + 0.01 / document.rank,
                    url=snippet.url,
                )
            )
    candidates.sort(key=lambda item: (-item.score, item.pmid, item.text))
    return candidates[:limit]
