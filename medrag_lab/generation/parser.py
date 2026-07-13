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
    return GeneratedAnswer.model_validate(json.loads(payload[start : end + 1]))
