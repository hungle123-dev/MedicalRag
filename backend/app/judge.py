from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

import httpx

from .env import load_dotenv


def blind_candidate(candidate_answer: str, evidence: list[dict]) -> tuple[str, list[dict]]:
    """Hide retrieval modality and provider-specific citation IDs from the judge."""
    mapping = {item["id"]: f"E{index}" for index, item in enumerate(evidence, start=1)}
    blinded_answer = candidate_answer
    for source, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
        blinded_answer = blinded_answer.replace(source, target)
    # Remove explicit modality cues without changing the medical content. This
    # is a second line of defence if a provider ignores the answer prompt.
    blinded_answer = re.sub(r"\b(?:PrimeKG|knowledge[ -]graph|structured evidence)\b",
                            "provided evidence", blinded_answer, flags=re.IGNORECASE)
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
    if not isinstance(parsed["claims"], list):
        raise ValueError("Judge claims must be a list")
    if not isinstance(parsed["justification"], str) or not parsed["justification"].strip():
        raise ValueError("Judge justification must be non-empty")
    if task == "correctness_completeness":
        for key in ("correctness", "completeness"):
            if type(parsed[key]) is not int or parsed[key] not in {0, 1, 2}:
                raise ValueError(f"Judge field {key} must be an integer in {{0,1,2}}")
    for key, (minimum, maximum) in numeric.items():
        if isinstance(parsed[key], bool) or not isinstance(parsed[key], (int, float)) or not minimum <= float(parsed[key]) <= maximum:
            raise ValueError(f"Judge field {key} is outside its rubric")


class GatewayJudge:
    """Two-pass blinded judge; never receives pipeline IDs or retrieval scores."""

    def __init__(self, root: Path, model: str = "cerebras/gpt-oss-120b"):
        load_dotenv(root)
        self.root, self.model = root, os.getenv("GATEWAY_JUDGE_MODEL", model)
        self.key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.futureppo.top/v1").rstrip("/")
        if not self.key: raise RuntimeError("OPENAI_API_KEY is not configured")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and
                                               parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise RuntimeError("OPENAI_BASE_URL must use HTTPS except for loopback development")
        self.cache = root / "artifacts/model_cache/gateway_judge"; self.cache.mkdir(parents=True, exist_ok=True)
        self._lock_guard, self._cache_locks = Lock(), {}

    def _cache_lock(self, digest: str) -> Lock:
        with self._lock_guard:
            return self._cache_locks.setdefault(digest, Lock())

    def evaluate(self, payload: dict) -> dict:
        prompt = (self.root / f"configs/prompts/judge_{payload['task']}_v1.txt").read_text(encoding="utf-8")
        content = prompt + "\n\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)
        digest = hashlib.sha256((self.base_url + self.model + content).encode()).hexdigest()
        target = self.cache / f"{digest}.json"
        with self._cache_lock(digest):
            if target.exists():
                cached = json.loads(target.read_text(encoding="utf-8"))
                parsed, raw = cached["parsed"], cached.get("raw", {})
                validate_judgement(payload["task"], parsed)
                return parsed | {"_judge": {"requested_model": self.model,
                                             "response_model": raw.get("model") or self.model,
                                             "system_fingerprint": raw.get("system_fingerprint"),
                                             "cached": True}}
            body = {"model": self.model, "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": content}]}
            last_error = None
            for attempt in range(3):
                try:
                    response = httpx.post(f"{self.base_url}/chat/completions",
                                          headers={"Authorization": f"Bearer {self.key}"},
                                          json=body, timeout=90)
                    response.raise_for_status()
                    raw = response.json()
                    parsed = json.loads(raw["choices"][0]["message"]["content"])
                    validate_judgement(payload["task"], parsed)
                    temporary = target.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps({"parsed": parsed, "raw": raw}), encoding="utf-8")
                    os.replace(temporary, target)
                    return parsed | {"_judge": {"requested_model": self.model,
                                                 "response_model": raw.get("model") or self.model,
                                                 "system_fingerprint": raw.get("system_fingerprint"),
                                                 "cached": False}}
                except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            raise RuntimeError(f"Judge failed after 3 attempts: {type(last_error).__name__}: {last_error}")
