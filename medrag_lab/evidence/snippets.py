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
    section: str = "abstract"
    begin: int | None = None
    end: int | None = None


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in SENTENCE.finditer(text):
        left, right = start, separator.start()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = separator.end()
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def sentence_windows(document: RetrievedDocument, size: int = 3) -> list[Snippet]:
    spans = _sentence_spans(document.text)
    if not spans:
        spans = [(0, len(document.text))]
    return [
        Snippet(
            pmid=document.pmid,
            title=document.title,
            text=document.text[spans[start][0] : spans[min(start + size, len(spans)) - 1][1]],
            score=document.score,
            url=document.url,
            begin=spans[start][0],
            end=spans[min(start + size, len(spans)) - 1][1],
        )
        for start in range(len(spans))
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
