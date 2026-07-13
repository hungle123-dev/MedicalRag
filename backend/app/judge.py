from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from .env import load_dotenv


def blind_candidate(candidate_answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Hide retrieval modality and provider-specific citation IDs from the judge."""
    mapping = {item["id"]: f"E{index}" for index, item in enumerate(evidence, start=1)}
    blinded_answer = candidate_answer
    for source, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
        blinded_answer = blinded_answer.replace(source, target)
    blinded_evidence = [{"id": mapping[item["id"]], "snippet": item.get("snippet", "")}
                        for item in evidence]
    return blinded_answer, blinded_evidence


def correctness_input(question: str, reference_answer: str, candidate_answer: str,
                      evidence: list[dict] | None = None) -> dict:
    if evidence is not None:
        candidate_answer, _ = blind_candidate(candidate_answer, evidence)
    return {"task": "correctness_completeness", "question": question,
            "reference_answer": reference_answer, "candidate_answer": candidate_answer}


def faithfulness_input(candidate_answer: str, cited_evidence: list[dict]) -> dict:
    candidate_answer, evidence = blind_candidate(candidate_answer, cited_evidence)
    return {"task": "faithfulness_citation", "candidate_answer": candidate_answer,
            "cited_evidence": evidence}


def validate_judgement(task: str, parsed: dict) -> None:
    numeric = ({"correctness": (0, 2), "completeness": (0, 2), "confidence": (0, 1)}
               if task == "correctness_completeness" else
               {"citation_precision": (0, 1), "citation_recall": (0, 1),
                "unsupported_claim_rate": (0, 1), "invented_citation_rate": (0, 1),
                "confidence": (0, 1)})
    required = {"claims", "justification", *numeric}
    if not isinstance(parsed, dict) or not required.issubset(parsed):
        keys = sorted(parsed) if isinstance(parsed, dict) else [type(parsed).__name__]
        raise ValueError(f"Judge response is missing required fields; keys={keys}")
    for key, (minimum, maximum) in numeric.items():
        if not isinstance(parsed[key], (int, float)) or not minimum <= float(parsed[key]) <= maximum:
            raise ValueError(f"Judge field {key} is outside its rubric")


class GatewayJudge:
    """Two-pass blinded judge; never receives pipeline IDs or retrieval scores."""

    def __init__(self, root: Path, model: str = "cerebras/gpt-oss-120b"):
        load_dotenv(root)
        self.root, self.model = root, os.getenv("GATEWAY_JUDGE_MODEL", model)
        self.key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.futureppo.top/v1").rstrip("/")
        if not self.key: raise RuntimeError("OPENAI_API_KEY is not configured")
        self.cache = root / "artifacts/model_cache/gateway_judge"; self.cache.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: dict) -> dict:
        prompt = (self.root / f"configs/prompts/judge_{payload['task']}_v1.txt").read_text(encoding="utf-8")
        content = prompt + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)
        digest = hashlib.sha256((self.model + content).encode()).hexdigest()
        target = self.cache / f"{digest}.json"
        if target.exists():
            parsed = json.loads(target.read_text(encoding="utf-8"))["parsed"]
            validate_judgement(payload["task"], parsed)
            return parsed
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": content}]}
        last_error = None
        for attempt in range(3):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions",
                                      headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=90)
                response.raise_for_status()
                raw = response.json()
                parsed = json.loads(raw["choices"][0]["message"]["content"])
                validate_judgement(payload["task"], parsed)
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"parsed": parsed, "raw": raw}), encoding="utf-8")
                os.replace(temporary, target)
                return parsed
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge failed after 3 attempts: {type(last_error).__name__}: {last_error}")
