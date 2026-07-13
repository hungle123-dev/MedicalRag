"""Freeze the full BioASQ eval ID order without using answer labels for tuning."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/raw/bioasq/eval.jsonl"
with source.open(encoding="utf-8") as stream:
    ids = [json.loads(line)["question_id"] for line in stream]
output = {"seed": 20260712, "method": "published_split_order", "split": "eval_locked",
          "definition": "Locked means labels and output scores are never used for tuning.",
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "ids": ids}
(ROOT / "data/manifests/bioasq_eval_question_ids.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8")
print(len(ids))
