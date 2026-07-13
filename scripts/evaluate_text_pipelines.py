"""Compare B1/B2/B3 retrieval on one frozen BioASQ dev population."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.medcpt import MedCPTIndex, MedCPTReranker, reciprocal_rank_fusion
from app.retrieval import BM25Index
from evaluate_bm25 import question_metrics
from analyze_retrieval import paired_bootstrap


def main() -> None:
    frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
    with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
        by_id = {row["question_id"]: row for line in stream if (row := json.loads(line))}
    bm25, dense, reranker = BM25Index.load(ROOT / "indexes/bm25_c0.pkl"), MedCPTIndex(ROOT / "indexes/medcpt"), MedCPTReranker()
    rows = []
    for question_id in frozen["ids"]:
        row, results = by_id[question_id], {}
        question = row["question"]
        started = time.perf_counter(); lexical50 = bm25.search(question, 50)
        results["B1"] = {"ranking": [item["id"] for item in lexical50[:10]],
                         "latency_ms": (time.perf_counter() - started) * 1000}
        started = time.perf_counter(); dense50 = dense.search(question, 50)
        results["B2"] = {"ranking": [item["id"] for item in dense50[:10]],
                         "latency_ms": (time.perf_counter() - started) * 1000}
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(lexical50, dense50, k=60)[:30]
        reranked = reranker.rerank(question, fused, k=10)
        results["B3"] = {"ranking": [item["id"] for item in reranked],
                         "latency_ms": (time.perf_counter() - started) * 1000}
        gold = set(map(str, row["relevant_passage_ids"]))
        snippet_pmids = [str(item.get("document", "")).rstrip("/").split("/")[-1] for item in row.get("snippets", [])]
        for result in results.values():
            result["metrics"] = question_metrics(result["ranking"], gold, snippet_pmids=snippet_pmids)
        rows.append({"question_id": question_id, "pipelines": results})
    metrics = {}
    for pipeline in ("B1", "B2", "B3"):
        values = [row["pipelines"][pipeline] for row in rows]
        metrics[pipeline] = {key: round(sum(value["metrics"][key] for value in values) / len(values), 6)
                             for key in values[0]["metrics"]}
        metrics[pipeline]["mean_latency_ms"] = round(sum(value["latency_ms"] for value in values) / len(values), 3)
    comparisons = {left: paired_bootstrap(
        [row["pipelines"][left]["metrics"]["recall_at_10"] for row in rows],
        [row["pipelines"]["B3"]["metrics"]["recall_at_10"] for row in rows], seed=20260712)
        for left in ("B1", "B2")}
    summary = {"run_id": "bioasq_dev_b1_b2_b3_20260712", "questions": len(rows),
               "metrics": metrics, "paired_recall_delta_B3_minus": comparisons}
    artifact = ROOT / "artifacts/experiments/bioasq" / summary["run_id"]
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "retrieval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    (ROOT / "data/manifests/bioasq_text_pipelines_dev.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
