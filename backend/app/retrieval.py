from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN.findall(text)]


class BM25Index:
    def __init__(self, model: BM25Okapi, documents: list[dict]):
        self.model = model
        self.documents = documents

    @classmethod
    def build(cls, corpus: Path) -> "BM25Index":
        documents = []
        with corpus.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                documents.append({
                    "id": str(row["id"]), "title": row.get("title", ""), "text": row["text"],
                    "url": row.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{row['id']}/",
                })
        return cls(BM25Okapi([tokenize(f"{row['title']} {row['text']}") for row in documents]), documents)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as stream:
            value = pickle.load(stream)  # local generated index; never accept an uploaded pickle
        if not isinstance(value, cls):
            raise ValueError("Unexpected BM25 index type")
        return value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def search(self, question: str, k: int = 8) -> list[dict]:
        scores = self.model.get_scores(tokenize(question))
        indexes = np.argpartition(scores, -k)[-k:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [
            self.documents[int(index)] | {"score": float(scores[index]), "retriever": "bm25"}
            for index in indexes
        ]
