from __future__ import annotations

import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import sha256, stable_hash
from medrag_lab.evaluation.retrieval import retrieval_metrics
from medrag_lab.indexing.bm25 import BM25Index, Recipe
from medrag_lab.settings import ROOT, settings
from medrag_lab.tracking.mlflow_tracking import log_artifact, tracked_run


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_bm25(recipe: Recipe, force: bool = False) -> Path:
    config = settings()
    path = config.medrag_index_dir / f"bm25-{recipe}.pkl"
    if path.exists() and not force:
        return path
    index = BM25Index.build(config.medrag_data_dir / "corpus.jsonl", recipe)
    index.save(path)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_bm25(
    recipe: Recipe = "title_abstract", population: str = "smoke40", limit: int | None = None
) -> dict[str, Any]:
    config = settings()
    split_path = ROOT / "data" / "manifests" / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError("Run `medrag data freeze` first")
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if population not in splits or not isinstance(splits[population], list):
        raise ValueError(f"Unknown population: {population}")
    allowed = set(map(str, splits[population]))
    dev_path = config.medrag_data_dir / "dev.jsonl"
    questions = [row for row in iter_jsonl(dev_path) if str(row["question_id"]) in allowed]
    questions.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]

    index_path = build_bm25(recipe)
    index = BM25Index.load(index_path)
    run_config = {
        "family": "E01",
        "arm": f"bm25_{recipe}",
        "population": population,
        "rows": len(questions),
        "candidate_depth": 100,
        "submit_depth": 10,
        "split_freeze_hash": splits["freeze_hash"],
        "corpus_sha256": sha256(config.medrag_data_dir / "corpus.jsonl"),
        "dev_sha256": sha256(dev_path),
        "git_sha": git_sha(),
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E01-{recipe}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    with tracked_run(run_name, run_config):
        for row in questions:
            try:
                documents, latency = index.search(str(row["question"]), 100)
                ranked = [document.pmid for document in documents]
                gold = set(map(str, row["relevant_passage_ids"]))
                top10, top100 = (
                    retrieval_metrics(ranked, gold, 10),
                    retrieval_metrics(ranked, gold, 100),
                )
                latencies.append(latency)
                predictions.append(
                    {
                        "question_id": str(row["question_id"]),
                        "question_type": str(row["type"]),
                        "ranked_pmids": ranked,
                        "latency_ms": latency,
                        "metrics": top10 | {"recall_at_100": top100["recall"]},
                        "failed": False,
                    }
                )
            except Exception as exc:  # intention-to-treat retains a zero-scored failure
                predictions.append(
                    {
                        "question_id": str(row["question_id"]),
                        "question_type": str(row["type"]),
                        "ranked_pmids": [],
                        "latency_ms": 0.0,
                        "metrics": dict.fromkeys(
                            ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100"), 0.0
                        ),
                        "failed": True,
                        "error_type": type(exc).__name__,
                    }
                )
        _write_jsonl(predictions_path, predictions)
        metric_names = ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100")
        aggregate = {
            name: statistics.fmean(row["metrics"][name] for row in predictions)
            for name in metric_names
        }
        failures = sum(row["failed"] for row in predictions)
        aggregate |= {
            "questions": len(predictions),
            "failures": failures,
            "failure_rate": failures / len(predictions) if predictions else 0.0,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
            "latency_ms_p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
            if latencies
            else 0.0,
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data",
            "scope": "closed-world positive-only gold-conditioned candidate pool",
            "config": run_config,
            "metrics": aggregate,
            "artifacts": {"predictions": str(predictions_path.relative_to(ROOT))},
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({key: float(value) for key, value in aggregate.items()})
        log_artifact(predictions_path)
        log_artifact(summary_path)
    return summary
