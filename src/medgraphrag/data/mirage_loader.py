"""Loads the real MIRAGE benchmark JSON (5 subtasks, 7663 questions).

File shape: {subtask: {qid: {"question": str, "options": {"A":...}, "answer": "A"}}}
Downloaded once via scripts/download_data.py into data/raw/mirage_benchmark.json.
"""
import json
from pathlib import Path

from medgraphrag.core.types import Question

SUBTASKS = ("medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu")


def load_mirage(path: str, subtask: str | None = None) -> list[Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    subtasks = [subtask] if subtask else list(raw.keys())
    out = []
    for st in subtasks:
        for qid, item in raw[st].items():
            out.append(Question(
                qid=f"{st}_{qid}",
                text=item["question"],
                options=item["options"],
                answer=item["answer"],
            ))
    return out
