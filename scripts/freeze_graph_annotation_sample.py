import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
output = {"source": "BioASQ dev frozen sample", "selection": "first 100 IDs from pre-output seeded random sample",
          "seed": frozen["seed"], "ids": frozen["ids"][:100], "labels_complete": False,
          "external_blocker": "qualified medical graph review"}
(ROOT / "data/manifests/graph_annotation_dev_ids.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(len(output["ids"]))
