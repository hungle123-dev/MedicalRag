from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .env import load_dotenv


CITATION = re.compile(r"\[([^\[\]]+)\]")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_evidence(evidence: list[dict]) -> str:
    return "\n\n".join(
        f"ID: {item['id']}\nTYPE: {item['type']}\nTITLE: {item.get('title', '')}\n"
        f"CONTENT: {item.get('snippet', '')}"
        for item in evidence
    )


def validate_citations(answer: str, evidence: list[dict]) -> dict:
    registry = {item["id"]: item for item in evidence}
    cited = list(dict.fromkeys(
        citation.strip() for group in CITATION.findall(answer) for citation in group.split(",")
        if citation.strip()
    ))
    valid = [citation for citation in cited if citation in registry]
    invented = [citation for citation in cited if citation not in registry]
    return {"valid_ids": valid, "invented_ids": invented, "valid": not invented}


@dataclass(frozen=True)
class Generation:
    answer: str
    model: str
    provider: str
    cached: bool = False
    response_model: str | None = None
    system_fingerprint: str | None = None
    usage: dict | None = None


class MockGenerator:
    model = "deterministic-evidence-mock-v1"
    provider = "local"

    def generate(self, question: str, evidence: list[dict], closed_book: bool = False) -> Generation:
        if not evidence:
            answer = "The supplied evidence is insufficient to answer this research question."
        else:
            first = evidence[0]
            answer = (
                f"Mock mode: medical synthesis is disabled; inspect the retrieved evidence [{first['id']}]. "
                "Configure a credentialed generator for an answer."
            )
        return Generation(answer=answer, model=self.model, provider=self.provider)


class GeminiGenerator:
    provider = "google"

    def __init__(self, root: Path, model: str = "gemini-3.5-flash"):
        self.root = root
        self.model = os.getenv("GEMINI_MODEL", model)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        self.prompt_template = (root / "configs/prompts/answer_v1.txt").read_text(encoding="utf-8")
        self.cache = root / "artifacts/model_cache/gemini"
        self.cache.mkdir(parents=True, exist_ok=True)

    def generate(self, question: str, evidence: list[dict], closed_book: bool = False) -> Generation:
        template = self.prompt_template
        if closed_book:
            template = (self.root / "configs/prompts/answer_closed_book_v1.txt").read_text(encoding="utf-8")
        prompt = template.format(question=question, evidence=render_evidence(evidence))
        key = stable_hash(json.dumps({"model": self.model, "prompt": prompt}, sort_keys=True))
        target = self.cache / f"{key}.json"
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
            return Generation(payload["answer"], self.model, self.provider, cached=True)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "topP": 1, "maxOutputTokens": 512},
        }
        last_error = None
        for attempt in range(3):
            try:
                response = httpx.post(url, params={"key": self.api_key}, json=body, timeout=90)
                response.raise_for_status()
                raw = response.json()
                answer = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"answer": answer, "raw": raw}), encoding="utf-8")
                os.replace(temporary, target)
                return Generation(answer, self.model, self.provider)
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini generation failed after 3 attempts: {type(last_error).__name__}")


class GatewayGenerator:
    provider = "futureppo"

    def __init__(self, root: Path, model: str | None = None):
        load_dotenv(root)
        self.root = root
        self.model = model or os.getenv("GATEWAY_GENERATOR_MODEL", "deepseek-v3.2")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.futureppo.top/v1").rstrip("/")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        if not self.base_url.startswith(("http://", "https://")):
            raise RuntimeError("OPENAI_BASE_URL must be an HTTP(S) URL")
        self.prompt_template = (root / "configs/prompts/answer_v1.txt").read_text(encoding="utf-8")
        self.cache = root / "artifacts/model_cache/gateway"
        self.cache.mkdir(parents=True, exist_ok=True)

    def generate(self, question: str, evidence: list[dict], closed_book: bool = False) -> Generation:
        template = self.prompt_template
        if closed_book:
            template = (self.root / "configs/prompts/answer_closed_book_v1.txt").read_text(encoding="utf-8")
        prompt = template.format(question=question, evidence=render_evidence(evidence))
        key = stable_hash(json.dumps({"model": self.model, "prompt": prompt}, sort_keys=True))
        target = self.cache / f"{key}.json"
        if target.exists():
            cached = json.loads(target.read_text(encoding="utf-8"))
            raw = cached.get("raw", {})
            return Generation(cached["answer"], self.model, self.provider, cached=True,
                              response_model=raw.get("model"), system_fingerprint=raw.get("system_fingerprint"),
                              usage=raw.get("usage"))
        body = {"model": self.model, "temperature": 0, "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}]}
        last_error = None
        for attempt in range(3):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions",
                                      headers={"Authorization": f"Bearer {self.api_key}"},
                                      json=body, timeout=90)
                response.raise_for_status()
                raw = response.json()
                answer = raw["choices"][0]["message"]["content"].strip()
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"answer": answer, "raw": raw}), encoding="utf-8")
                os.replace(temporary, target)
                return Generation(answer, self.model, self.provider, response_model=raw.get("model"),
                                  system_fingerprint=raw.get("system_fingerprint"), usage=raw.get("usage"))
            except (httpx.HTTPError, KeyError, IndexError, AttributeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Gateway generation failed after 3 attempts: {type(last_error).__name__}")


def create_generator(root: Path):
    load_dotenv(root)
    selected = os.getenv("MEDICAL_RAG_GENERATOR", "mock").casefold()
    if selected == "mock":
        return MockGenerator()
    if selected == "gemini":
        return GeminiGenerator(root)
    if selected == "gateway":
        return GatewayGenerator(root)
    raise RuntimeError(f"Unsupported generator provider: {selected}")
