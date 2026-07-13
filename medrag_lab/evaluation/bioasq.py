from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

TOKEN = re.compile(r"[\w]+(?:[-_/][\w]+)*", re.UNICODE)


def normalize_answer(value: str) -> str:
    return " ".join(TOKEN.findall(value.casefold()))


def exact_answer_score(
    question_type: str, prediction: str | list[str] | None, gold: object
) -> dict[str, float]:
    if question_type == "summary":
        return {"enabled": 0.0}
    if gold is None:
        raise ValueError("Official exact gold is unavailable; exact scoring is disabled")
    if question_type == "yesno":
        expected = normalize_answer(str(gold))
        actual = normalize_answer(str(prediction or ""))
        return {"accuracy": float(actual == expected)}

    predicted = [str(prediction)] if isinstance(prediction, str) else list(prediction or [])
    gold_entities = gold if isinstance(gold, list) else [gold]
    aliases: list[set[str]] = []
    for entity in gold_entities:
        values = entity if isinstance(entity, list) else [entity]
        aliases.append({normalize_answer(str(value)) for value in values})

    if question_type == "factoid":
        first_rank = next(
            (
                rank
                for rank, value in enumerate(predicted[:5], 1)
                if any(normalize_answer(value) in entity for entity in aliases)
            ),
            None,
        )
        return {
            "strict_accuracy": float(
                bool(predicted) and normalize_answer(predicted[0]) in aliases[0]
            ),
            "lenient_accuracy": float(first_rank is not None),
            "mrr": 1.0 / first_rank if first_rank else 0.0,
        }

    predicted_set = {normalize_answer(value) for value in predicted[:100]}
    matched = sum(bool(predicted_set & entity) for entity in aliases)
    precision = matched / len(predicted_set) if predicted_set else 0.0
    recall = matched / len(aliases) if aliases else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _su4_units(text: str) -> Counter[tuple[str, ...]]:
    tokens = [token.casefold() for token in TOKEN.findall(text)]
    units: Counter[tuple[str, ...]] = Counter((token,) for token in tokens)
    for left in range(len(tokens)):
        for right in range(left + 1, min(len(tokens), left + 6)):
            units[(tokens[left], tokens[right])] += 1
    return units


def rouge_su4(prediction: str, reference: str) -> dict[str, float]:
    predicted, expected = _su4_units(prediction), _su4_units(reference)
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def snippet_span_f1(predicted: Sequence[dict], gold: Sequence[dict]) -> dict[str, float]:
    def positions(snippets: Sequence[dict]) -> set[tuple[str, str, int]]:
        result: set[tuple[str, str, int]] = set()
        for snippet in snippets:
            document = str(snippet.get("document", "")).rstrip("/").rsplit("/", 1)[-1]
            section = str(snippet.get("beginSection", "")).casefold()
            begin, end = snippet.get("offsetInBeginSection"), snippet.get("offsetInEndSection")
            if isinstance(begin, int) and isinstance(end, int) and 0 <= begin <= end:
                result.update((document, section, offset) for offset in range(begin, end))
        return result

    predicted_positions, gold_positions = positions(predicted), positions(gold)
    overlap = len(predicted_positions & gold_positions)
    precision = overlap / len(predicted_positions) if predicted_positions else 0.0
    recall = overlap / len(gold_positions) if gold_positions else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
