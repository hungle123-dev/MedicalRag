from __future__ import annotations

import tiktoken

from medrag_lab.evidence.snippets import Snippet, sentence_windows
from medrag_lab.schemas import RetrievedDocument

ENCODING = tiktoken.get_encoding("o200k_base")


def chunk_documents(documents: list[RetrievedDocument], sentence_window: int = 3) -> list[Snippet]:
    """Create sentence-window evidence chunks while preserving PMID provenance."""
    if not 1 <= sentence_window <= 8:
        raise ValueError("sentence_window must be between 1 and 8")
    return [
        chunk for document in documents for chunk in sentence_windows(document, sentence_window)
    ]


def fixed_token_chunks(
    documents: list[RetrievedDocument], size: int = 256, overlap: int = 64
) -> list[Snippet]:
    if size < 32 or not 0 <= overlap < size:
        raise ValueError("size must be >= 32 and overlap must be in [0, size)")
    chunks: list[Snippet] = []
    stride = size - overlap
    for document in documents:
        tokens = ENCODING.encode(document.text)
        search_from = 0
        for start in range(0, len(tokens), stride):
            text = ENCODING.decode(tokens[start : start + size]).strip()
            if text:
                begin = document.text.find(text, max(0, search_from - overlap * 4))
                if begin < 0:
                    begin = document.text.find(text)
                end = begin + len(text) if begin >= 0 else None
                chunks.append(
                    Snippet(
                        pmid=document.pmid,
                        title=document.title,
                        text=text,
                        score=document.score,
                        url=document.url,
                        begin=begin if begin >= 0 else None,
                        end=end,
                    )
                )
                if begin >= 0:
                    search_from = begin
            if start + size >= len(tokens):
                break
    return chunks
