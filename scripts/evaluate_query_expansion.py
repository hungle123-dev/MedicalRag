"""Paired Q0/Q1 BioASQ experiment using deployable PrimeKG entity/type expansion."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.graph import PrimeKGIndex
from app.retrieval import BM25Index
from evaluate_bm25 import question_metrics
from analyze_retrieval import paired_bootstrap


def main() -> None:
    frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
    with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
        by_id = {row["question_id"]: row for line in stream if (row := json.loads(line))}
    text = BM25Index.load(ROOT / "indexes/bm25_c0.pkl")
    graph = PrimeKGIndex(ROOT / "indexes/primekg.sqlite3")
    rows = []
    for question_id in frozen["ids"]:
        row = by_id[question_id]; original = row["question"]
        seeds = graph.link(original)
        suffix = " ".join(f"{seed['name']} {seed['type'].replace('/', ' ')}" for seed in seeds)
        expanded = f"{original} {suffix}".strip()
        variants = []
        for strategy, query in (("Q0", original), ("Q1", expanded)):
            started = time.perf_counter(); ranking = [item["id"] for item in text.search(query, 10)]
            metrics = question_metrics(ranking, set(map(str, row["relevant_passage_ids"])))
            variants.append({"strategy": strategy, "query": query, "ranking": ranking, "metrics": metrics,
                             "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
        delta = variants[1]["metrics"]["recall_at_10"] - variants[0]["metrics"]["recall_at_10"]
        rows.append({"question_id": question_id, "linked_entities": [seed["name"] for seed in seeds],
                     "variants": variants, "rescued": delta > 0, "harmed": delta < 0})
    q0 = [row["variants"][0]["metrics"]["recall_at_10"] for row in rows]
    q1 = [row["variants"][1]["metrics"]["recall_at_10"] for row in rows]
    summary = {"run_id": "bioasq_dev_q1_bm25_20260712", "questions": len(rows),
        "linked_query_rate": round(sum(bool(row["linked_entities"]) for row in rows) / len(rows), 6),
        "recall_at_10": {"Q0": round(sum(q0) / len(q0), 6), "Q1": round(sum(q1) / len(q1), 6)},
        "paired_delta": paired_bootstrap(q0, q1, seed=20260712),
        "rescued_query_rate": round(sum(row["rescued"] for row in rows) / len(rows), 6),
        "harmed_query_rate": round(sum(row["harmed"] for row in rows) / len(rows), 6),
        "mean_latency_ms": {name: round(sum(row["variants"][index]["latency_ms"] for row in rows) / len(rows), 3)
                            for index, name in enumerate(("Q0", "Q1"))}}
    artifact = ROOT / "artifacts/experiments/bioasq" / summary["run_id"]
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "retrieval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    (ROOT / "data/manifests/bioasq_query_expansion_dev.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
