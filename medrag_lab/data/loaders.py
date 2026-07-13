from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from medrag_lab.data.schemas import GoldQuestion, InferenceQuestion


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def load_gold_questions(path: Path) -> list[GoldQuestion]:
    return [GoldQuestion.model_validate(row) for row in iter_jsonl(path)]


def load_inference_questions(
    path: Path, allowed_ids: set[str] | None = None
) -> list[InferenceQuestion]:
    """Create a gold-free deployable view at the data boundary."""
    questions: list[InferenceQuestion] = []
    for row in iter_jsonl(path):
        question_id = str(row["question_id"])
        if allowed_ids is None or question_id in allowed_ids:
            questions.append(
                InferenceQuestion(question_id=question_id, question=str(row["question"]))
            )
    return questions
