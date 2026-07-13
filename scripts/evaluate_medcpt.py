"""Evaluate C0/C2 MedCPT retrieval on the same frozen BioASQ dev questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.medcpt import MedCPTIndex
from evaluate_bm25 import question_metrics


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def collapse(rows: list[dict], k: int = 10) -> list[str]:
    ranking, seen = [], set()
    for row in rows:
        if row["id"] not in seen:
            ranking.append(row["id"]); seen.add(row["id"])
        if len(ranking) == k: break
    return ranking


def evaluate(strategy: str, index_path: Path, questions: list[dict]) -> dict:
    index = MedCPTIndex(index_path)
    started = time.perf_counter(); rows = []
    for question in questions:
        ranking = collapse(index.search(question["question"], k=100), 10)
        snippet_pmids = [str(item.get("document", "")).rstrip("/").split("/")[-1]
                         for item in question.get("snippets", [])]
        metrics = question_metrics(ranking, set(map(str, question["relevant_passage_ids"])),
                                   snippet_pmids=snippet_pmids)
        rows.append({"question_id": question["question_id"], "ranking": ranking, "metrics": metrics})
    aggregate = {key: round(sum(row["metrics"][key] for row in rows) / len(rows), 6)
                 for key in rows[0]["metrics"]}
    return {"strategy": strategy, "retriever": "MedCPT", "documents": index.index.ntotal,
            "questions": len(rows), "metrics": aggregate,
            "elapsed_seconds": round(time.perf_counter() - started, 3), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0", type=Path, default=ROOT / "indexes/medcpt")
    parser.add_argument("--c2", type=Path, default=ROOT / "indexes/medcpt_c2")
    args = parser.parse_args()
    frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
    by_id = {row["question_id"]: row for row in load_jsonl(ROOT / "data/raw/bioasq/dev.jsonl")}
    questions = [by_id[question_id] for question_id in frozen["ids"]]
    results = [evaluate("C0", args.c0, questions), evaluate("C2", args.c2, questions)]
    run_id = "bioasq_dev_medcpt_" + hashlib.sha256(json.dumps(frozen["ids"]).encode()).hexdigest()[:12]
    artifact = ROOT / "artifacts/experiments/bioasq" / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"run_id": run_id, "question_ids_file": "data/manifests/bioasq_dev_question_ids.json",
               "results": [{key: value for key, value in result.items() if key != "rows"} for result in results]}
    (ROOT / "data/manifests/bioasq_medcpt_dev.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
