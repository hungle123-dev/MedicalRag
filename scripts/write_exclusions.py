import json
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "data/manifests/exclusions.jsonl"
with output.open("w", encoding="utf-8") as target:
    for split in ("train", "val", "test"):
        path = ROOT / f"data/raw/primekgqa/{split}_call_bioLLM.json"
        with path.open("rb") as stream:
            for row in ijson.items(stream, "item"):
                if not str(row.get("generated_question") or "").strip():
                    target.write(json.dumps({"dataset": "PrimeKGQA", "split": split, "id": row.get("id"),
                                             "reason": "missing_natural_language_question"}) + "\n")
print(sum(1 for _ in output.open(encoding="utf-8")))
