from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from medrag_lab.generation.parser import parse_generated_answer
from medrag_lab.generation.schemas import GatewayResult
from medrag_lab.settings import ROOT, Settings, settings


class GatewayClient:
    def __init__(self, config: Settings | None = None):
        self.config = config or settings()
        if not self.config.openai_api_key or not self.config.openai_base_url:
            raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
        self.client = OpenAI(
            api_key=self.config.openai_api_key.get_secret_value(),
            base_url=self.config.openai_base_url.rstrip("/"),
            timeout=90,
            max_retries=0,
        )
        self.cache_dir = ROOT / "artifacts" / "cache" / "gateway"

    def list_models(self) -> list[str]:
        return sorted(model.id for model in self.client.models.list().data)

    def snapshot_models(self, path: Path | None = None) -> Path:
        models = self.list_models()
        destination = path or ROOT / "artifacts" / "gateway" / "model_inventory.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "base_url": self.config.openai_base_url,
            "models": models,
        }
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_output_tokens: int = 800,
    ) -> GatewayResult:
        selected_model = model or self.config.gateway_generator_model
        if not selected_model:
            raise ValueError("GATEWAY_GENERATOR_MODEL is required")
        payload = {
            "model": selected_model,
            "system": system_prompt,
            "user": user_prompt,
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
        }
        cache_path = self._cache_path(payload)
        if cache_path.is_file():
            cached = GatewayResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
            return cached.model_copy(update={"cached": True})

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempts in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=selected_model,
                    temperature=0,
                    max_tokens=max_output_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                answer = parse_generated_answer(content)
                break
            except Exception as exc:
                last_error = exc
                if attempts == 3:
                    raise
                time.sleep(2 ** (attempts - 1))
        else:  # pragma: no cover - loop either breaks or raises
            raise RuntimeError("Gateway retry loop exhausted") from last_error
        usage = response.usage
        result = GatewayResult(
            answer=answer,
            model=str(response.model or selected_model),
            provider=self.config.openai_base_url,
            input_tokens=int(usage.prompt_tokens if usage else 0),
            output_tokens=int(usage.completion_tokens if usage else 0),
            latency_ms=(time.perf_counter() - started) * 1_000,
            attempts=attempts,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(cache_path)
        return result
