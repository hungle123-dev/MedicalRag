from __future__ import annotations

import json
import re

from medrag_lab.generation.schemas import GeneratedAnswer

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_generated_answer(text: str) -> GeneratedAnswer:
    candidate = FENCE.search(text)
    payload = candidate.group(1) if candidate else text
    start, end = payload.find("{"), payload.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Provider response did not contain a JSON object")
    value = json.loads(payload[start : end + 1])
    if (
        isinstance(value, dict)
        and value.get("abstained") is True
        and not str(value.get("ideal_answer") or "").strip()
    ):
        # Providers commonly emit null/empty text for a valid abstention. Normalize only
        # this explicitly signalled case; unsupported non-abstaining answers remain errors.
        value["ideal_answer"] = "Insufficient evidence in the provided context."
    return GeneratedAnswer.model_validate(value)
