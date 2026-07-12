"""Loads MedRAG textbook chunks (real MedCorp subset) into a flat {id: text} corpus.

Files: data/raw/medrag_textbooks/chunk/*.jsonl, one JSON object per line:
{"id": "<book>_<n>", "title": "<book>", "content": "<chunk text>"}
"""
import json
from pathlib import Path


def load_textbook_corpus(dir_path: str, books: list[str] | None = None,
                          limit: int | None = None) -> dict[str, str]:
    root = Path(dir_path)
    files = sorted(root.glob("*.jsonl"))
    if books:
        wanted = set(books)
        files = [f for f in files if f.stem in wanted]
    corpus: dict[str, str] = {}
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(corpus) >= limit:
                    return corpus
                obj = json.loads(line)
                corpus[obj["id"]] = obj["content"]
    return corpus
