"""Measure exact entity-link and accepted-path coverage on the frozen dev population."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipelines import graph_evidence

frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
ids = set(frozen["ids"][:300])
with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
    rows = [row for line in stream if (row := json.loads(line)) and row["question_id"] in ids]

records = []
for row in rows:
    paths, seeds = graph_evidence(row["question"])
    records.append({"question_id": row["question_id"], "question_type": row["type"],
                    "linked_entities": len(seeds), "accepted_paths": len(paths),
                    "entity_types": sorted({seed["type"] for seed in seeds}),
                    "hop_counts": sorted({path["hop_count"] for path in paths})})

by_type = {}
for kind in sorted({row["question_type"] for row in records}):
    selected = [row for row in records if row["question_type"] == kind]
    by_type[kind] = {"n": len(selected),
                     "any_entity_link_rate": round(sum(row["linked_entities"] > 0 for row in selected) / len(selected), 6),
                     "accepted_path_rate": round(sum(row["accepted_paths"] > 0 for row in selected) / len(selected), 6)}
report = {
    "run_id": "bioasq_dev_graph_coverage_20260713", "questions": len(records),
    "entity_linker": "exact longest PrimeKG name with type-cue tiebreak",
    "path_acceptance_threshold": 0.8, "max_hops": 2,
    "any_entity_link_rate": round(sum(row["linked_entities"] > 0 for row in records) / len(records), 6),
    "accepted_path_rate": round(sum(row["accepted_paths"] > 0 for row in records) / len(records), 6),
    "mean_linked_entities": round(sum(row["linked_entities"] for row in records) / len(records), 3),
    "mean_accepted_paths": round(sum(row["accepted_paths"] for row in records) / len(records), 3),
    "linked_entity_types": dict(Counter(kind for row in records for kind in row["entity_types"])),
    "by_question_type": by_type,
    "interpretation": "Retrieval coverage only; this is not a gold entity-link accuracy or graph-answerability label.",
}
(ROOT / "data/manifests/bioasq_graph_coverage_dev.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
