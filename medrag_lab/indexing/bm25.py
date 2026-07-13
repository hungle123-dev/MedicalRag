from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.schemas import RetrievedDocument

Recipe = Literal["title", "abstract", "title_abstract", "boosted_title_abstract_mesh"]
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN.findall(text)]


def document_text(row: dict[str, Any], recipe: Recipe) -> str:
    title, abstract = str(row.get("title", "")), str(row.get("text", ""))
    if recipe == "title":
        return title
    if recipe == "abstract":
        return abstract
    if recipe == "title_abstract":
        return f"{title} {abstract}"
    mesh = " ".join(map(str, row.get("mesh_terms", [])))
    return f"{title} {title} {abstract} {mesh}"


class BM25Index:
    def __init__(self, model: BM25Okapi, documents: list[dict[str, str]], recipe: Recipe):
        self.model = model
        self.documents = documents
        self.recipe = recipe

    @classmethod
    def build(cls, corpus: Path, recipe: Recipe = "title_abstract") -> BM25Index:
        documents: list[dict[str, str]] = []
        tokenized: list[list[str]] = []
        for row in iter_jsonl(corpus):
            pmid = str(row["id"])
            documents.append(
                {
                    "pmid": pmid,
                    "title": str(row.get("title", "")),
                    "text": str(row.get("text", "")),
                    "url": str(row.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                }
            )
            tokenized.append(tokenize(document_text(row, recipe)))
        return cls(BM25Okapi(tokenized), documents, recipe)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        # Generated local indexes only; this is never called on user-uploaded files.
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, cls):
            raise ValueError("Unexpected BM25 index payload")
        return value

    def search(self, query: str, k: int = 10) -> tuple[list[RetrievedDocument], float]:
        if not 1 <= k <= len(self.documents):
            raise ValueError("k must be within corpus size")
        started = time.perf_counter()
        scores = np.asarray(self.model.get_scores(tokenize(query)))
        indexes = np.argpartition(scores, -k)[-k:] if k < len(scores) else np.arange(len(scores))
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        documents = [
            RetrievedDocument(
                **self.documents[int(index)],
                score=float(scores[index]),
                rank=rank,
                retriever=f"bm25:{self.recipe}",
            )
            for rank, index in enumerate(indexes, 1)
        ]
        return documents, (time.perf_counter() - started) * 1_000
