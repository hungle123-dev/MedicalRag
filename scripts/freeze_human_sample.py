"""Freeze the blinded BioASQ human-review population before model outputs exist."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260712

with (ROOT / "data/raw/bioasq/eval.jsonl").open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream]
sample = random.Random(SEED).sample(rows, 100)
output = {"seed": SEED, "method": "simple_random_without_replacement", "split": "eval_locked",
          "frozen_before_output_review": True, "ids": [row["question_id"] for row in sample]}
(ROOT / "data/manifests/bioasq_human_100.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
