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
from medrag_lab.data.splits import normalize_question
from medrag_lab.evaluation.bioasq import rouge_su4
from medrag_lab.evaluation.retrieval import retrieval_metrics
from medrag_lab.evaluation.statistics import (
    paired_effect_size,
    paired_group_bootstrap,
    paired_permutation_p,
)
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
        aggregate["map_at_10"] = aggregate["ap"]
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


def run_generation(
    pipeline_id: str, population: str = "smoke40", limit: int | None = None
) -> dict[str, Any]:
    """Run gold-free inference first, then score the sealed prediction file separately."""
    from medrag_lab.data.loaders import load_inference_questions
    from medrag_lab.pipeline import MedicalRAGPipeline, load_pipeline_config
    from medrag_lab.schemas import AnswerRequest

    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    if population not in splits or not isinstance(splits[population], list):
        raise ValueError(f"Unknown population: {population}")
    allowed = set(map(str, splits[population]))
    questions = load_inference_questions(config.medrag_data_dir / "dev.jsonl", allowed)
    questions.sort(key=lambda row: row.question_id)
    if limit is not None:
        questions = questions[:limit]
    pipeline_config = load_pipeline_config(pipeline_id)
    run_config = {
        "family": "E08",
        "arm": pipeline_id,
        "population": population,
        "rows": len(questions),
        "split_freeze_hash": splits["freeze_hash"],
        "generator_model": pipeline_config["generator_model"] or settings().gateway_generator_model,
        "pipeline_config_hash": stable_hash(pipeline_config),
        "git_sha": git_sha(),
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E08-{pipeline_id}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    inference_path = output_dir / "inference.jsonl"
    scored_path = output_dir / "scored.jsonl"
    summary_path = output_dir / "summary.json"
    inference: list[dict[str, Any]] = []
    pipeline = MedicalRAGPipeline(pipeline_id)
    with tracked_run(run_name, run_config):
        for question in questions:
            try:
                answer = pipeline.answer(
                    AnswerRequest(question=question.question, pipeline_id=pipeline_id)
                )
                inference.append(
                    {
                        "question_id": question.question_id,
                        "answer": answer.model_dump(),
                        "failed": False,
                    }
                )
            except Exception as exc:
                inference.append(
                    {
                        "question_id": question.question_id,
                        "answer": None,
                        "failed": True,
                        "error_type": type(exc).__name__,
                    }
                )
        _write_jsonl(inference_path, inference)

        # Evaluation boundary: gold is loaded only after the inference artifact is closed.
        by_id = {
            str(row["question_id"]): row
            for row in iter_jsonl(config.medrag_data_dir / "dev.jsonl")
            if str(row["question_id"]) in {item["question_id"] for item in inference}
        }
        scored = []
        for item in inference:
            prediction = item["answer"]["ideal_answer"] if item["answer"] else ""
            metric = rouge_su4(prediction, str(by_id[item["question_id"]]["answer"]))
            scored.append(item | {"rouge_su4": metric})
        _write_jsonl(scored_path, scored)
        f1_values = [row["rouge_su4"]["f1"] for row in scored]
        failures = sum(row["failed"] for row in scored)
        latencies = [row["answer"]["latency_ms"] for row in scored if row["answer"]]
        input_tokens = sum(row["answer"]["input_tokens"] for row in scored if row["answer"])
        output_tokens = sum(row["answer"]["output_tokens"] for row in scored if row["answer"])
        aggregate = {
            "rouge_su4_f1": statistics.fmean(f1_values) if f1_values else 0,
            "questions": len(scored),
            "failures": failures,
            "failure_rate": failures / len(scored) if scored else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0,
            "latency_ms_p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
            if latencies
            else 0,
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data_real_gateway",
            "scope": "closed-world positive-only gold-conditioned candidate pool",
            "config": run_config,
            "metrics": aggregate,
            "artifacts": {
                "inference": str(inference_path.relative_to(ROOT)),
                "scored": str(scored_path.relative_to(ROOT)),
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({key: float(value) for key, value in aggregate.items()})
        log_artifact(inference_path)
        log_artifact(scored_path)
        log_artifact(summary_path)
    return summary


def run_dense_retrieval(
    method: str = "medcpt", population: str = "smoke40", limit: int | None = None
) -> dict[str, Any]:
    if method not in {"medcpt", "rrf", "rrf_rerank"}:
        raise ValueError("method must be medcpt, rrf, or rrf_rerank")
    from medrag_lab.retrieval.dense import MedCPTRetriever
    from medrag_lab.retrieval.hybrid import reciprocal_rank_fusion

    config = settings()
    split_path = ROOT / "data" / "manifests" / "splits.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if population not in splits or not isinstance(splits[population], list):
        raise ValueError(f"Unknown population: {population}")
    allowed = set(map(str, splits[population]))
    questions = [
        row
        for row in iter_jsonl(config.medrag_data_dir / "dev.jsonl")
        if str(row["question_id"]) in allowed
    ]
    questions.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        questions = questions[:limit]

    dense = MedCPTRetriever()
    sparse = BM25Index.load(build_bm25("title_abstract")) if method != "medcpt" else None
    reranker = None
    if method == "rrf_rerank":
        from medrag_lab.retrieval.reranker import MedCPTReranker

        reranker = MedCPTReranker()
    metadata = json.loads(dense.paths.metadata.read_text(encoding="utf-8"))
    run_config = {
        "family": "E02",
        "arm": method,
        "population": population,
        "rows": len(questions),
        "candidate_depth": 100,
        "split_freeze_hash": splits["freeze_hash"],
        "corpus_sha256": metadata["corpus_sha256"],
        "article_revision": metadata["article_revision"],
        "query_revision": metadata["query_revision"],
        "git_sha": git_sha(),
    }
    if reranker:
        run_config["cross_encoder_revision"] = reranker.revision
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E02-{method}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    with tracked_run(run_name, run_config):
        for row in questions:
            try:
                dense_rows, latency = dense.retrieve(str(row["question"]), 100)
                ranked_rows = dense_rows
                if sparse:
                    sparse_rows, sparse_ms = sparse.search(str(row["question"]), 100)
                    latency += sparse_ms
                    ranked_rows = reciprocal_rank_fusion(sparse_rows, dense_rows)[:100]
                if reranker:
                    ranked_rows, rerank_ms = reranker.rerank(str(row["question"]), ranked_rows, 100)
                    latency += rerank_ms
                ranked = [item.pmid for item in ranked_rows]
                gold = set(map(str, row["relevant_passage_ids"]))
                top10 = retrieval_metrics(ranked, gold, 10)
                top100 = retrieval_metrics(ranked, gold, 100)
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
            except Exception as exc:
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
        names = ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100")
        aggregate = {
            name: statistics.fmean(row["metrics"][name] for row in predictions) for name in names
        }
        aggregate["map_at_10"] = aggregate["ap"]
        failures = sum(row["failed"] for row in predictions)
        aggregate |= {
            "questions": len(predictions),
            "failures": failures,
            "failure_rate": failures / len(predictions) if predictions else 0,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0,
            "latency_ms_p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
            if latencies
            else 0,
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


def run_oracle(population: str = "validation200", limit: int | None = None) -> dict[str, Any]:
    """L0-L3 bottleneck localization; gold evidence is passed only as an explicit override."""
    import re

    from medrag_lab.evidence.snippets import Snippet
    from medrag_lab.generation.prompts import CLOSED_BOOK_SYSTEM_PROMPT
    from medrag_lab.pipeline import MedicalRAGPipeline
    from medrag_lab.schemas import AnswerRequest, RetrievedDocument

    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    allowed = set(map(str, splits[population]))
    gold_rows = [
        row
        for row in iter_jsonl(config.medrag_data_dir / "dev.jsonl")
        if str(row["question_id"]) in allowed
    ]
    gold_rows.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        gold_rows = gold_rows[:limit]
    corpus = {str(row["id"]): row for row in iter_jsonl(config.medrag_data_dir / "corpus.jsonl")}
    pipeline_id = "bm25_deepseek"
    pipeline = MedicalRAGPipeline(pipeline_id)
    run_config = {
        "family": "E10",
        "population": population,
        "rows": len(gold_rows),
        "pipeline": pipeline_id,
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E10-oracle-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    arms = ("L0_closed_book", "L1_predicted", "L2_gold_documents", "L3_gold_snippets")
    rows: list[dict[str, Any]] = []
    with tracked_run(run_name, run_config):
        for row in gold_rows:
            pmids = list(dict.fromkeys(map(str, row["relevant_passage_ids"])))
            gold_documents = [
                RetrievedDocument(
                    pmid=pmid,
                    title=str(corpus[pmid].get("title", "")),
                    text=str(corpus[pmid].get("text", "")),
                    url=str(corpus[pmid].get("url", "")),
                    score=1,
                    rank=rank,
                    retriever="gold_document_oracle",
                )
                for rank, pmid in enumerate(pmids, 1)
                if pmid in corpus
            ]
            gold_snippets = []
            for snippet in row["snippets"]:
                match = re.search(r"(\d+)/?$", str(snippet.get("document", "")))
                pmid = match.group(1) if match else ""
                if pmid and str(snippet.get("text", "")).strip():
                    source = corpus.get(pmid, {})
                    gold_snippets.append(
                        Snippet(
                            pmid=pmid,
                            title=str(source.get("title", "")),
                            text=str(snippet["text"]),
                            score=1,
                            url=str(source.get("url", snippet.get("document", ""))),
                        )
                    )
            request = AnswerRequest(question=str(row["question"]), pipeline_id=pipeline_id)
            for arm in arms:
                try:
                    if arm == "L0_closed_book":
                        answer = pipeline.answer(
                            request,
                            evidence_override=[],
                            system_prompt_override=CLOSED_BOOK_SYSTEM_PROMPT,
                        )
                    elif arm == "L1_predicted":
                        answer = pipeline.answer(request)
                    elif arm == "L2_gold_documents":
                        answer = pipeline.answer(request, documents_override=gold_documents)
                    else:
                        answer = pipeline.answer(request, evidence_override=gold_snippets)
                    metric = rouge_su4(answer.ideal_answer, str(row["answer"]))
                    rows.append(
                        {
                            "question_id": str(row["question_id"]),
                            "question_type": str(row["type"]),
                            "arm": arm,
                            "answer": answer.model_dump(),
                            "rouge_su4": metric,
                            "failed": False,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "question_id": str(row["question_id"]),
                            "question_type": str(row["type"]),
                            "arm": arm,
                            "answer": None,
                            "rouge_su4": {"precision": 0, "recall": 0, "f1": 0},
                            "failed": True,
                            "error_type": type(exc).__name__,
                        }
                    )
        predictions_path = output_dir / "oracle_predictions.jsonl"
        _write_jsonl(predictions_path, rows)
        metrics = {
            arm: {
                "rouge_su4_f1": statistics.fmean(
                    item["rouge_su4"]["f1"] for item in rows if item["arm"] == arm
                ),
                "failures": sum(item["failed"] for item in rows if item["arm"] == arm),
            }
            for arm in arms
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data_real_gateway",
            "config": run_config,
            "metrics": metrics,
            "interpretation": "Diagnostic upper bounds, not deployable systems",
            "artifacts": {"predictions": str(predictions_path.relative_to(ROOT))},
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics(
            {f"{arm}.rouge_su4_f1": value["rouge_su4_f1"] for arm, value in metrics.items()}
        )
        log_artifact(predictions_path)
        log_artifact(summary_path)
    return summary


def compare_prediction_files(
    left_path: Path, right_path: Path, metric: str = "metrics.ap"
) -> dict[str, Any]:
    left = {str(row["question_id"]): row for row in iter_jsonl(left_path)}
    right = {str(row["question_id"]): row for row in iter_jsonl(right_path)}
    if set(left) != set(right):
        raise ValueError("Paired comparison requires identical question IDs")
    questions = {}
    for filename in ("dev.jsonl", "eval.jsonl"):
        for row in iter_jsonl(settings().medrag_data_dir / filename):
            question_id = str(row["question_id"])
            if question_id in left:
                questions[question_id] = normalize_question(str(row["question"]))
    if set(questions) != set(left):
        raise ValueError("Could not resolve all normalized question groups")

    def value(row: dict[str, Any]) -> float:
        current: Any = row
        for key in metric.split("."):
            current = current[key]
        return float(current)

    identifiers = sorted(left)
    left_values = [value(left[item]) for item in identifiers]
    right_values = [value(right[item]) for item in identifiers]
    groups = [questions[item] for item in identifiers]
    result = {
        "left": str(left_path),
        "right": str(right_path),
        "metric": metric,
        "rows": len(identifiers),
        "bootstrap": paired_group_bootstrap(left_values, right_values, groups),
        "paired_effect_size": paired_effect_size(left_values, right_values),
        "paired_permutation_p": paired_permutation_p(left_values, right_values),
    }
    comparison_hash = stable_hash(result)
    result["comparison_hash"] = comparison_hash
    destination = ROOT / "reports" / "comparisons" / f"{comparison_hash[:12]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
