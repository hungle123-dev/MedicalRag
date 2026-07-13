"""Verify the frozen gateway model IDs without printing credentials."""
import json
import os
import sys
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))
from app.env import load_dotenv

load_dotenv(root)
key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.futureppo.top/v1").rstrip("/")
report = {}
if key:
    response = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    response.raise_for_status()
    models = {item["id"] for item in response.json().get("data", [])}
    for role, target in {
        "generator": os.getenv("GATEWAY_GENERATOR_MODEL", "deepseek-v3.2"),
        "judge": os.getenv("GATEWAY_JUDGE_MODEL", "cerebras/gpt-oss-120b"),
    }.items():
        report[role] = {"target": target, "available": target in models}
else:
    report["gateway"] = {"available": False, "blocker": "OPENAI_API_KEY missing"}
(root / "data/manifests/llm_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
