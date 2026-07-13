from __future__ import annotations


def original_query(question: str) -> list[str]:
    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("Question cannot be empty")
    return [cleaned]
