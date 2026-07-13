"""Verify exact generator/judge IDs without printing credentials."""
import json
import os
from pathlib import Path

import httpx

report = {}
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    response = httpx.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": gemini_key}, timeout=30); response.raise_for_status()
    models = {item["name"].removeprefix("models/") for item in response.json().get("models", [])}
    target = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    report["gemini"] = {"target": target, "available": target in models}
else: report["gemini"] = {"available": False, "blocker": "GEMINI_API_KEY missing"}
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    response = httpx.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {groq_key}"}, timeout=30); response.raise_for_status()
    models = {item["id"] for item in response.json().get("data", [])}
    target = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b")
    report["groq_judge"] = {"target": target, "available": target in models}
else: report["groq_judge"] = {"available": False, "blocker": "GROQ_API_KEY missing"}
root = Path(__file__).resolve().parents[1]
(root / "data/manifests/llm_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
