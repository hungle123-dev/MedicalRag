from __future__ import annotations

import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import sha256, stable_hash
from medrag_lab.data.splits import normalize_question
from medrag_lab.evaluation.bioasq import rouge_su4
from medrag_lab.evaluation.retrieval import retrieval_metrics
from medrag_lab.evaluation.statistics import (
    nearest_rank_percentile,
    paired_effect_size,
    paired_group_bootstrap,
    paired_mde_80,
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
        "purpose": "feasibility_only"
        if population == "smoke40" or limit is not None
        else "candidate_evaluation",
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
            "latency_ms_p95": nearest_rank_percentile(latencies, 0.95) if latencies else 0.0,
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
    from medrag_lab.generation.prompts import SYSTEM_PROMPT
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
        "provider": urlparse(settings().openai_base_url).netloc,
        "pipeline_config_hash": stable_hash(pipeline_config),
        "system_prompt_hash": stable_hash(SYSTEM_PROMPT),
        "git_sha": git_sha(),
        "purpose": "feasibility_only"
        if population == "smoke40" or limit is not None
        else "candidate_evaluation",
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
                trace = pipeline.trace_store.get(answer.trace_id) or {}
                inference.append(
                    {
                        "question_id": question.question_id,
                        "answer": answer.model_dump(),
                        "evidence_hash": stable_hash(trace.get("serialized_context", "")),
                        "prompt_hash": trace.get("prompt_hash", ""),
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
            "latency_ms_p95": nearest_rank_percentile(latencies, 0.95) if latencies else 0,
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
    method: str = "medcpt",
    population: str = "smoke40",
    limit: int | None = None,
    bm25_recipe: Recipe = "title_abstract",
    offset: int = 0,
    rerank_batch_size: int = 64,
    serial_latency: bool = False,
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
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if rerank_batch_size < 1:
        raise ValueError("rerank_batch_size must be positive")
    questions = questions[offset : offset + limit if limit is not None else None]

    dense = MedCPTRetriever()
    sparse = BM25Index.load(build_bm25(bm25_recipe)) if method != "medcpt" else None
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
        "offset": offset,
        "candidate_depth": 100,
        "latency_mode": "dedicated_serial" if serial_latency else "batched_throughput_amortized",
        "warmup_queries": 1 if serial_latency and questions else 0,
        "bm25_recipe": bm25_recipe if method != "medcpt" else "not_applicable",
        "split_freeze_hash": splits["freeze_hash"],
        "corpus_sha256": metadata["corpus_sha256"],
        "article_revision": metadata["article_revision"],
        "query_revision": metadata["query_revision"],
        "git_sha": git_sha(),
        "purpose": (
            "latency_benchmark"
            if serial_latency
            else "feasibility_only"
            if population == "smoke40"
            else "candidate_shard"
            if limit is not None or offset
            else "candidate_evaluation"
        ),
    }
    if reranker:
        run_config["cross_encoder_revision"] = reranker.revision
        run_config["rerank_batch_size"] = rerank_batch_size
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E02-{method}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    if serial_latency and questions:
        warmup_query = str(questions[0]["question"])
        warmup_dense, _ = dense.retrieve(warmup_query, 100)
        if reranker and sparse:
            warmup_sparse, _ = sparse.search(warmup_query, 100)
            warmup_hybrid = reciprocal_rank_fusion(warmup_sparse, warmup_dense)[:100]
            reranker.rerank(warmup_query, warmup_hybrid, 100, batch_size=rerank_batch_size)
    dense_results = (
        [dense.retrieve(str(row["question"]), 100) for row in questions]
        if serial_latency
        else dense.retrieve_many([str(row["question"]) for row in questions], 100)
    )
    with tracked_run(run_name, run_config):
        for row, (dense_rows, dense_latency) in zip(questions, dense_results, strict=True):
            try:
                latency = dense_latency
                ranked_rows = dense_rows
                if sparse:
                    sparse_rows, sparse_ms = sparse.search(str(row["question"]), 100)
                    latency += sparse_ms
                    ranked_rows = reciprocal_rank_fusion(sparse_rows, dense_rows)[:100]
                if reranker:
                    ranked_rows, rerank_ms = reranker.rerank(
                        str(row["question"]),
                        ranked_rows,
                        100,
                        batch_size=rerank_batch_size,
                    )
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
            "latency_ms_p95": nearest_rank_percentile(latencies, 0.95) if latencies else 0,
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


def run_oracle(
    population: str = "validation200",
    limit: int | None = None,
    pipeline_id: str = "bm25_deepseek",
) -> dict[str, Any]:
    """L0-L3 bottleneck localization; gold evidence is passed only as an explicit override."""
    import re

    from medrag_lab.evidence.snippets import Snippet
    from medrag_lab.generation.prompts import CLOSED_BOOK_SYSTEM_PROMPT
    from medrag_lab.pipeline import MedicalRAGPipeline
    from medrag_lab.schemas import AnswerRequest, RetrievedDocument

    config = settings()
    if population == "heldout340":
        from medrag_lab.experiments.final import verify_final_freeze

        verify_final_freeze()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    allowed = set(map(str, splits[population]))
    question_source = config.medrag_data_dir / (
        "eval.jsonl" if population == "heldout340" else "dev.jsonl"
    )
    gold_rows = [row for row in iter_jsonl(question_source) if str(row["question_id"]) in allowed]
    gold_rows.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        gold_rows = gold_rows[:limit]
    corpus = {str(row["id"]): row for row in iter_jsonl(config.medrag_data_dir / "corpus.jsonl")}
    pipeline = MedicalRAGPipeline(pipeline_id)
    run_config = {
        "family": "E10",
        "population": population,
        "rows": len(gold_rows),
        "pipeline": pipeline_id,
        "document_cap": 10,
        "snippet_cap": int(pipeline.config["snippet_limit"]),
        "context_token_budget": int(pipeline.config["context_token_budget"]),
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "feasibility_only"
        if population == "smoke40" or limit is not None
        else "diagnostic_evaluation",
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E10-oracle-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    arms = ("L0_closed_book", "L1_predicted", "L2_gold_documents", "L3_gold_snippets")
    rows: list[dict[str, Any]] = []
    with tracked_run(run_name, run_config):
        for row in gold_rows:
            pmids = list(dict.fromkeys(map(str, row["relevant_passage_ids"])))[:10]
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
            gold_snippets = gold_snippets[: int(pipeline.config["snippet_limit"])]
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
        if population == "heldout340":
            from medrag_lab.experiments.final import record_heldout_access

            record_heldout_access("E10.gold_evidence_oracle", predictions_path)
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
    left_path: Path,
    right_path: Path,
    metric: str = "metrics.ap",
    *,
    require_equal_evidence: bool = False,
) -> dict[str, Any]:
    left = {str(row["question_id"]): row for row in iter_jsonl(left_path)}
    right = {str(row["question_id"]): row for row in iter_jsonl(right_path)}
    if set(left) != set(right):
        raise ValueError("Paired comparison requires identical question IDs")
    shared_hash_rows = [
        question_id
        for question_id in left
        if "evidence_hash" in left[question_id] and "evidence_hash" in right[question_id]
    ]
    if require_equal_evidence and len(shared_hash_rows) != len(left):
        raise ValueError("Equal-evidence comparison requires an evidence hash for every row")
    if require_equal_evidence and any(
        left[item]["evidence_hash"] != right[item]["evidence_hash"] for item in shared_hash_rows
    ):
        raise ValueError("Generator comparison evidence hashes are not identical")
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
        "require_equal_evidence": require_equal_evidence,
        "rows": len(identifiers),
        "bootstrap": paired_group_bootstrap(left_values, right_values, groups),
        "paired_effect_size": paired_effect_size(left_values, right_values),
        "paired_permutation_p": paired_permutation_p(left_values, right_values),
        "normal_approx_mde_80": paired_mde_80(left_values, right_values, groups),
    }
    comparison_hash = stable_hash(result)
    result["comparison_hash"] = comparison_hash
    destination = ROOT / "reports" / "comparisons" / f"{comparison_hash[:12]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_judge_sanity() -> dict[str, Any]:
    from medrag_lab.evaluation.llm_panel import LLMPanel

    source = next(
        row
        for row in iter_jsonl(settings().medrag_data_dir / "corpus.jsonl")
        if str(row["id"]) == "20711702"
    )
    question = "Do BRCA1 mutations confer inherited breast cancer risk?"
    reference = "Yes. Pathogenic germline BRCA1 mutations increase inherited breast cancer risk."
    evidence = f"[PMID:20711702] {source['title']}\n{source['text']}"
    good = reference
    bad = "No. BRCA1 mutations eliminate breast cancer risk and make counseling unnecessary."
    panel = LLMPanel()
    good_result = panel.direct(question, good, reference, evidence)
    bad_result = panel.direct(question, bad, reference, evidence)
    pairwise = panel.pairwise("sanity-brca", question, good, bad, evidence)
    passed = (
        good_result["median_correctness_0_3"] > bad_result["median_correctness_0_3"]
        and good_result["median_completeness_0_3"] >= bad_result["median_completeness_0_3"]
        and good_result["median_evidence_faithfulness_0_3"]
        > bad_result["median_evidence_faithfulness_0_3"]
        and good_result["median_unsupported_atomic_claim_rate"]
        <= bad_result["median_unsupported_atomic_claim_rate"]
        and pairwise["panel_winner"] == "left"
    )
    result = {
        "status": "observed_real_gateway_sanity",
        "fixture_pmid": "20711702",
        "models": [entry["model"] for entry in good_result["judges"]],
        "good": good_result,
        "adversarial": bad_result,
        "pairwise_position_swapped": pairwise,
        "passed": passed,
        "limitation": "Automated proxy sanity check; not human or physician validation",
    }
    destination = ROOT / "reports" / "judge_sanity.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_superiority_gate(
    comparison_path: Path,
    left_efficiency_summary: Path,
    right_efficiency_summary: Path,
    gate_id: str,
) -> dict[str, Any]:
    from medrag_lab.experiments.gates import superiority_gate

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    left_summary = json.loads(left_efficiency_summary.read_text(encoding="utf-8"))
    right_summary = json.loads(right_efficiency_summary.read_text(encoding="utf-8"))
    latency_modes = {
        value.get("config", {}).get("latency_mode") for value in (left_summary, right_summary)
    }
    if None not in latency_modes and latency_modes != {"dedicated_serial"}:
        raise ValueError("Latency gate requires dedicated serial efficiency runs")
    left, right = left_summary["metrics"], right_summary["metrics"]
    bootstrap = comparison["bootstrap"]
    result = {
        "gate_id": gate_id,
        "quality_comparison_hash": comparison["comparison_hash"],
        "efficiency_runs": [str(left_efficiency_summary), str(right_efficiency_summary)],
        **superiority_gate(
            bootstrap["mean_delta_right_minus_left"],
            bootstrap["ci95_low"],
            right["failure_rate"] - left["failure_rate"],
            right["latency_ms_p95"] / left["latency_ms_p95"],
        ),
    }
    result["gate_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "gates" / f"{gate_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_query_retrieval(
    strategy: str,
    population: str = "query800",
    limit: int | None = None,
    bm25_recipe: Recipe = "boosted_title_abstract_mesh",
    retriever: str = "rrf",
    workers: int = 4,
    query_model: str = "gemini-2.5-flash-lite",
) -> dict[str, Any]:
    if strategy not in {"original", "mesh", "hyde"}:
        raise ValueError("strategy must be original, mesh, or hyde")
    if retriever not in {"rrf", "rrf_rerank"}:
        raise ValueError("retriever must be rrf or rrf_rerank")
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    from medrag_lab.generation.gateway import GatewayClient
    from medrag_lab.query.hyde import HyDEExpander
    from medrag_lab.query.mesh import MeshExpander
    from medrag_lab.retrieval.dense import MedCPTRetriever
    from medrag_lab.retrieval.hybrid import reciprocal_rank_fusion

    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text(encoding="utf-8"))
    allowed = set(map(str, splits[population]))
    questions = [
        row
        for row in iter_jsonl(config.medrag_data_dir / "dev.jsonl")
        if str(row["question_id"]) in allowed
    ]
    questions.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        questions = questions[:limit]
    sparse = BM25Index.load(build_bm25(bm25_recipe))
    dense = MedCPTRetriever()
    mesh = MeshExpander(config.medrag_data_dir / "corpus.jsonl") if strategy == "mesh" else None
    hyde = HyDEExpander(GatewayClient(), model=query_model) if strategy == "hyde" else None
    reranker = None
    if retriever == "rrf_rerank":
        from medrag_lab.retrieval.reranker import MedCPTReranker

        reranker = MedCPTReranker()
    run_config = {
        "family": "E03",
        "arm": strategy,
        "population": population,
        "rows": len(questions),
        "retriever_control": retriever,
        "workers": workers,
        "bm25_recipe": bm25_recipe,
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "feasibility_only" if limit is not None else "candidate_evaluation",
    }
    if hyde:
        from medrag_lab.query.hyde import HYDE_SYSTEM

        run_config |= {
            "query_generator_model": query_model,
            "hyde_prompt_hash": stable_hash(HYDE_SYSTEM),
        }
    if reranker:
        run_config["cross_encoder_revision"] = reranker.revision
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E03-{strategy}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    prediction_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    predictions = []
    latencies = []
    transform_latencies: list[float] = []
    transform_cached = [False] * len(questions)
    transform_input_tokens = [0] * len(questions)
    transform_output_tokens = [0] * len(questions)
    transformed: list[str | None] = [None] * len(questions)
    transform_errors: dict[int, str] = {}
    if strategy == "original":
        transformed = [str(row["question"]) for row in questions]
        transform_latencies = [0.0] * len(questions)
    elif mesh:
        for index, row in enumerate(questions):
            started = time.perf_counter()
            transformed[index] = mesh.expand(str(row["question"]))[0]
            transform_latencies.append((time.perf_counter() - started) * 1_000)
    elif hyde:

        def expand(index: int) -> tuple[int, str, float, bool, int, int]:
            value = hyde.expand(str(questions[index]["question"]))
            return (
                index,
                value.expanded,
                value.latency_ms,
                value.cached,
                value.input_tokens,
                value.output_tokens,
            )

        measured = [0.0] * len(questions)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(expand, index): index for index in range(len(questions))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, value, latency, cached, input_tokens, output_tokens = future.result()
                    transformed[index] = value
                    measured[index] = latency
                    transform_cached[index] = cached
                    transform_input_tokens[index] = input_tokens
                    transform_output_tokens[index] = output_tokens
                except Exception as exc:
                    transform_errors[index] = type(exc).__name__
        transform_latencies = measured
    dense_results = dense.retrieve_many([value or "" for value in transformed], 100)
    with tracked_run(run_name, run_config):
        for index, (row, dense_result) in enumerate(zip(questions, dense_results, strict=True)):
            try:
                query_value = transformed[index]
                if index in transform_errors or query_value is None:
                    raise RuntimeError(transform_errors.get(index, "QueryTransformationFailure"))
                query = query_value
                sparse_rows, sparse_ms = sparse.search(query, 100)
                dense_rows, dense_ms = dense_result
                ranked_rows = reciprocal_rank_fusion(sparse_rows, dense_rows)[:100]
                rerank_ms = 0.0
                if reranker:
                    ranked_rows, rerank_ms = reranker.rerank(query, ranked_rows, 100)
                ranked = [item.pmid for item in ranked_rows]
                latency = sparse_ms + dense_ms + rerank_ms + transform_latencies[index]
                gold = set(map(str, row["relevant_passage_ids"]))
                top10 = retrieval_metrics(ranked, gold, 10)
                top100 = retrieval_metrics(ranked, gold, 100)
                latencies.append(latency)
                predictions.append(
                    {
                        "question_id": str(row["question_id"]),
                        "question_type": str(row["type"]),
                        "transformed_query_hash": stable_hash(query),
                        "ranked_pmids": ranked,
                        "latency_ms": latency,
                        "query_transform_ms": transform_latencies[index],
                        "query_transform_cached": transform_cached[index],
                        "metrics": top10 | {"recall_at_100": top100["recall"]},
                        "failed": False,
                    }
                )
            except Exception as exc:
                predictions.append(
                    {
                        "question_id": str(row["question_id"]),
                        "ranked_pmids": [],
                        "latency_ms": 0,
                        "metrics": dict.fromkeys(
                            ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100"), 0.0
                        ),
                        "failed": True,
                        "error_type": type(exc).__name__,
                    }
                )
        _write_jsonl(prediction_path, predictions)
        names = ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100")
        metrics = {
            name: statistics.fmean(item["metrics"][name] for item in predictions) for name in names
        }
        metrics["map_at_10"] = metrics["ap"]
        failures = sum(item["failed"] for item in predictions)
        metrics |= {
            "questions": len(predictions),
            "failures": failures,
            "failure_rate": failures / len(predictions) if predictions else 0,
            "retrieval_latency_ms_p50": statistics.median(latencies) if latencies else 0,
            "retrieval_latency_ms_p95": nearest_rank_percentile(latencies, 0.95)
            if latencies
            else 0,
            "query_transform_ms_p50": statistics.median(transform_latencies)
            if transform_latencies
            else 0,
            "query_transform_cache_rate": statistics.fmean(map(float, transform_cached))
            if transform_cached
            else 0,
            "query_transform_input_tokens": sum(transform_input_tokens),
            "query_transform_output_tokens": sum(transform_output_tokens),
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data",
            "scope": "closed-world positive-only gold-conditioned candidate pool",
            "config": run_config,
            "metrics": metrics,
            "artifacts": {"predictions": str(prediction_path.relative_to(ROOT))},
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_artifact(prediction_path)
        log_artifact(summary_path)
    return summary


def subset_retrieval_predictions(
    source_path: Path,
    population: str,
    family: str,
    arm: str,
) -> dict[str, Any]:
    """Reuse a frozen superset prediction artifact without rerunning an unchanged baseline."""
    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    rows = [row for row in iter_jsonl(source_path) if str(row["question_id"]) in allowed]
    rows.sort(key=lambda row: str(row["question_id"]))
    if {str(row["question_id"]) for row in rows} != allowed:
        raise ValueError("Source prediction file does not cover the requested population")
    run_config = {
        "family": family,
        "arm": arm,
        "population": population,
        "rows": len(rows),
        "source_prediction_sha256": sha256(source_path),
        "reuse_policy": "exact_frozen_superset_subset_no_reinference",
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "candidate_evaluation",
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"{family}-{arm}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    _write_jsonl(predictions_path, rows)
    names = ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100")
    metrics = {
        name: statistics.fmean(float(row["metrics"][name]) for row in rows) for name in names
    }
    metrics["map_at_10"] = metrics["ap"]
    failures = sum(bool(row.get("failed")) for row in rows)
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    metrics |= {
        "questions": len(rows),
        "failures": failures,
        "failure_rate": failures / len(rows) if rows else 0.0,
        "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
        "latency_ms_p95": nearest_rank_percentile(latencies, 0.95) if latencies else 0.0,
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_real_data_reused_predictions",
        "scope": "closed-world positive-only gold-conditioned candidate pool",
        "config": run_config,
        "metrics": metrics,
        "artifacts": {"predictions": str(predictions_path.relative_to(ROOT))},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def merge_retrieval_shards(
    source_paths: list[Path], population: str, family: str, arm: str
) -> dict[str, Any]:
    """Merge resumable shards, preferring a successful retry over a failed first attempt."""
    if not source_paths:
        raise ValueError("At least one source shard is required")
    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    selected: dict[str, dict[str, Any]] = {}
    recovered = 0
    for path in source_paths:
        for row in iter_jsonl(path):
            question_id = str(row["question_id"])
            if question_id not in allowed:
                continue
            previous = selected.get(question_id)
            if previous is None:
                selected[question_id] = row
            elif previous.get("failed") and not row.get("failed"):
                selected[question_id] = row
                recovered += 1
            elif (
                not previous.get("failed")
                and not row.get("failed")
                and previous.get("ranked_pmids") != row.get("ranked_pmids")
            ):
                raise ValueError(f"Conflicting successful predictions for {question_id}")
    if set(selected) != allowed:
        raise ValueError(f"Merged shards are missing {len(allowed - set(selected))} question IDs")
    rows = [selected[item] for item in sorted(selected)]
    remaining_failures = sum(bool(row.get("failed")) for row in rows)
    if remaining_failures:
        raise ValueError(f"Merged shards still contain {remaining_failures} failed questions")
    run_config = {
        "family": family,
        "arm": arm,
        "population": population,
        "rows": len(rows),
        "merge_policy": "successful_retry_replaces_failed_attempt",
        "recovered_failures": recovered,
        "source_sha256": [sha256(path) for path in source_paths],
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "candidate_evaluation",
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"{family}-{arm}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    _write_jsonl(predictions_path, rows)
    names = ("ap", "recall", "mrr", "ndcg", "hit", "recall_at_100")
    metrics = {
        name: statistics.fmean(float(row["metrics"][name]) for row in rows) for name in names
    }
    metrics["map_at_10"] = metrics["ap"]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    metrics |= {
        "questions": len(rows),
        "failures": 0,
        "failure_rate": 0.0,
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": nearest_rank_percentile(latencies, 0.95),
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_real_data_recovered_shards",
        "scope": "closed-world positive-only gold-conditioned candidate pool",
        "config": run_config,
        "metrics": metrics,
        "artifacts": {"predictions": str(predictions_path.relative_to(ROOT))},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
