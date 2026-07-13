from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from .env import load_dotenv


def correctness_input(question: str, reference_answer: str, candidate_answer: str) -> dict:
    return {"task": "correctness_completeness", "question": question,
            "reference_answer": reference_answer, "candidate_answer": candidate_answer}


def faithfulness_input(candidate_answer: str, cited_evidence: list[dict]) -> dict:
    allowed = {"id", "type", "title", "snippet", "source", "pmid", "provenance"}
    return {"task": "faithfulness_citation", "candidate_answer": candidate_answer,
            "cited_evidence": [{key: value for key, value in item.items() if key in allowed}
                               for item in cited_evidence]}


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
        if target.exists(): return json.loads(target.read_text(encoding="utf-8"))["parsed"]
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": content}]}
        for attempt in range(3):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions",
                                      headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=90)
                response.raise_for_status(); break
            except httpx.HTTPError:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
        raw = response.json(); parsed = json.loads(raw["choices"][0]["message"]["content"])
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"parsed": parsed, "raw": raw}), encoding="utf-8")
        os.replace(temporary, target)
        return parsed
