from __future__ import annotations

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import mlflow

from medrag_lab.data.loaders import iter_jsonl, load_inference_questions
from medrag_lab.data.manifests import sha256, stable_hash
from medrag_lab.evaluation.bioasq import rouge_su4
from medrag_lab.evaluation.semantic import bertscore, rouge2
from medrag_lab.evaluation.statistics import nearest_rank_percentile
from medrag_lab.evidence.snippets import Snippet
from medrag_lab.experiments.runner import _write_jsonl, git_sha
from medrag_lab.generation.gateway import GatewayClient
from medrag_lab.generation.prompts import (
    CITATION_SYSTEM_PROMPT,
    CLOSED_BOOK_SYSTEM_PROMPT,
    GENERIC_STRUCTURED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    answer_prompt,
    prompt_hash,
)
from medrag_lab.pipeline import MedicalRAGPipeline
from medrag_lab.schemas import RetrievedDocument
from medrag_lab.settings import ROOT, settings
from medrag_lab.tracking.mlflow_tracking import log_artifact, tracked_run

PromptStyle = Literal[
    "generic_structured",
    "citation_constraint",
    "predicted_type_schema",
    "gold_type_oracle",
    "closed_book",
]
PMID_LABEL = re.compile(r"\[PMID:(\d+)\]")


def _question_source(population: str) -> Path:
    filename = "eval.jsonl" if population == "heldout340" else "dev.jsonl"
    return settings().medrag_data_dir / filename


def _guard_heldout(population: str) -> None:
    if population == "heldout340":
        from medrag_lab.experiments.final import verify_final_freeze

        verify_final_freeze()


def _prompt(style: PromptStyle, gold_type: str | None = None) -> str:
    if style == "closed_book":
        return CLOSED_BOOK_SYSTEM_PROMPT
    if style == "gold_type_oracle":
        if gold_type is None:
            return SYSTEM_PROMPT + "\nThe benchmark-provided question type is {GOLD_TYPE}."
        return SYSTEM_PROMPT + f"\nThe benchmark-provided question type is {gold_type}."
    return {
        "generic_structured": GENERIC_STRUCTURED_SYSTEM_PROMPT,
        "citation_constraint": CITATION_SYSTEM_PROMPT,
        "predicted_type_schema": SYSTEM_PROMPT,
    }[style]


def _frozen_snippets(
    annotations: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]
) -> list[Snippet]:
    snippets = []
    for rank, annotation in enumerate(annotations, 1):
        pmid = str(annotation["document"]).rstrip("/").rsplit("/", 1)[-1]
        if pmid not in corpus:
            raise ValueError(f"Evidence PMID {pmid} is not in the frozen corpus")
        snippets.append(
            Snippet(
                pmid=pmid,
                title=str(corpus[pmid].get("title", "")),
                text=str(annotation["text"]),
                score=float(len(annotations) - rank + 1),
                url=str(corpus[pmid].get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                section=str(annotation.get("beginSection", "abstract")),
                begin=annotation.get("offsetInBeginSection"),
                end=annotation.get("offsetInEndSection"),
            )
        )
    return snippets


def prepare_contexts(
    family: str,
    arm: str,
    pipeline_id: str,
    population: str,
    *,
    context_token_budget: int = 1_200,
    context_order: str = "relevance_descending",
    diversity: str = "none",
    evidence_strategy: str = "sentence3",
    retrieval_predictions: Path | None = None,
    evidence_predictions: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Create a sealed gold-free context artifact that multiple generation arms can reuse."""
    config = settings()
    _guard_heldout(population)
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    questions = load_inference_questions(_question_source(population), allowed)
    questions.sort(key=lambda row: row.question_id)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    overrides = {
        "context_token_budget": context_token_budget,
        "context_order": context_order,
        "diversity": diversity,
        "evidence_strategy": evidence_strategy,
    }
    runtime_overrides = overrides | (
        {"retriever": "bm25"} if retrieval_predictions or evidence_predictions else {}
    )
    pipeline = MedicalRAGPipeline(pipeline_id, config_override=runtime_overrides)
    frozen_retrieval: dict[str, dict[str, Any]] = {}
    frozen_evidence: dict[str, dict[str, Any]] = {}
    corpus: dict[str, dict[str, Any]] = {}
    if retrieval_predictions is not None:
        frozen_retrieval = {
            str(row["question_id"]): row
            for row in iter_jsonl(retrieval_predictions)
            if str(row["question_id"]) in {question.question_id for question in questions}
        }
        if set(frozen_retrieval) != {question.question_id for question in questions}:
            raise ValueError("Frozen retrieval artifact does not cover context questions exactly")
    if evidence_predictions is not None:
        frozen_evidence = {
            str(row["question_id"]): row
            for row in iter_jsonl(evidence_predictions)
            if str(row["question_id"]) in {question.question_id for question in questions}
        }
        if set(frozen_evidence) != {question.question_id for question in questions}:
            raise ValueError("Frozen evidence artifact does not cover context questions exactly")
    if frozen_retrieval or frozen_evidence:
        corpus = {
            str(row["id"]): row for row in iter_jsonl(config.medrag_data_dir / "corpus.jsonl")
        }
    run_config = {
        "family": family,
        "arm": arm,
        "stage": "gold_free_context_preparation",
        "pipeline": pipeline_id,
        "population": population,
        "rows": len(questions),
        "overrides": overrides,
        "retrieval_source_sha256": sha256(retrieval_predictions)
        if retrieval_predictions
        else "live_pipeline",
        "evidence_source_sha256": sha256(evidence_predictions)
        if evidence_predictions
        else "live_pipeline",
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "feasibility_only" if limit is not None else "candidate_evaluation",
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"{family}-{arm}-contexts-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    contexts_path = output_dir / "contexts.jsonl"
    scored_path = output_dir / "scored_contexts.jsonl"
    summary_path = output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    with tracked_run(run_name, run_config):
        for question in questions:
            try:
                documents_override = None
                evidence_override = None
                if frozen_retrieval:
                    pmids = list(
                        map(
                            str,
                            frozen_retrieval[question.question_id]["ranked_pmids"][
                                : int(pipeline.config["retrieval_k"])
                            ],
                        )
                    )
                    documents_override = [
                        RetrievedDocument(
                            pmid=pmid,
                            title=str(corpus[pmid].get("title", "")),
                            text=str(corpus[pmid].get("text", "")),
                            url=str(
                                corpus[pmid].get("url")
                                or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            ),
                            score=1.0 / rank,
                            rank=rank,
                            retriever="frozen_prediction_artifact",
                        )
                        for rank, pmid in enumerate(pmids, 1)
                        if pmid in corpus
                    ]
                if frozen_evidence:
                    annotations = list(frozen_evidence[question.question_id]["snippets"])
                    evidence_override = _frozen_snippets(annotations, corpus)
                prepared = pipeline.prepare_context(
                    question.question,
                    documents_override=documents_override,
                    evidence_override=evidence_override,
                )
                evidence_set = sorted(
                    (item.pmid, item.section, item.begin, item.end, item.text)
                    for item in prepared.packed
                )
                candidate_set = sorted(
                    (item.pmid, item.section, item.begin, item.end, item.text)
                    for item in prepared.snippets
                )
                rows.append(
                    {
                        "question_id": question.question_id,
                        "question": question.question,
                        "context": prepared.serialized_context,
                        "context_hash": stable_hash(prepared.serialized_context),
                        "evidence_set_hash": stable_hash(evidence_set),
                        "candidate_evidence_hash": stable_hash(candidate_set),
                        "packed_evidence": [vars(item) for item in prepared.packed],
                        "retrieved_pmids": [item.pmid for item in prepared.documents],
                        "retrieval_ms": prepared.retrieval_ms,
                        "failed": False,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "question_id": question.question_id,
                        "question": question.question,
                        "context": "",
                        "context_hash": stable_hash(""),
                        "evidence_set_hash": stable_hash([]),
                        "candidate_evidence_hash": stable_hash([]),
                        "packed_evidence": [],
                        "retrieved_pmids": [],
                        "retrieval_ms": 0.0,
                        "failed": True,
                        "error_type": type(exc).__name__,
                    }
                )
        _write_jsonl(contexts_path, rows)
        scored_rows: list[dict[str, Any]] = []
        if population != "heldout340":
            gold = {
                str(row["question_id"]): set(map(str, row["relevant_passage_ids"]))
                for row in iter_jsonl(_question_source(population))
                if str(row["question_id"]) in {item["question_id"] for item in rows}
            }
            for row in rows:
                expected_pmids = gold[row["question_id"]]
                packed_pmids = {str(item["pmid"]) for item in row["packed_evidence"]}
                scored_rows.append(
                    row
                    | {
                        "packed_context_gold_pmid_recall": len(packed_pmids & expected_pmids)
                        / len(expected_pmids)
                        if expected_pmids
                        else 0.0
                    }
                )
            _write_jsonl(scored_path, scored_rows)
        failures = sum(row["failed"] for row in rows)
        metrics = {
            "questions": len(rows),
            "failures": failures,
            "failure_rate": failures / len(rows) if rows else 0.0,
            "packed_snippets_mean": statistics.fmean(len(row["packed_evidence"]) for row in rows)
            if rows
            else 0.0,
        }
        if scored_rows:
            metrics["packed_context_gold_pmid_recall"] = statistics.fmean(
                row["packed_context_gold_pmid_recall"] for row in scored_rows
            )
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data_gold_free",
            "config": run_config,
            "metrics": metrics,
            "artifacts": {
                "contexts": str(contexts_path.relative_to(ROOT)),
                **({"scored_contexts": str(scored_path.relative_to(ROOT))} if scored_rows else {}),
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_artifact(contexts_path)
        if scored_rows:
            log_artifact(scored_path)
        log_artifact(summary_path)
    return summary


def run_context_generation(
    family: str,
    arm: str,
    contexts_path: Path,
    population: str,
    model: str,
    prompt_style: PromptStyle = "predicted_type_schema",
    *,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate from a sealed context file, then open gold in a separate scoring phase."""
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    config = settings()
    _guard_heldout(population)
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    expected = sorted(allowed)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        expected = expected[:limit]
    expected_ids = set(expected)
    contexts = [row for row in iter_jsonl(contexts_path) if str(row["question_id"]) in expected_ids]
    contexts.sort(key=lambda row: str(row["question_id"]))
    if {str(row["question_id"]) for row in contexts} != expected_ids:
        raise ValueError("Context artifact does not cover the requested population")
    gold_types: dict[str, str] = {}
    if prompt_style == "gold_type_oracle":
        gold_types = {
            str(row["question_id"]): str(row["type"])
            for row in iter_jsonl(_question_source(population))
            if str(row["question_id"]) in expected_ids
        }
        if set(gold_types) != expected_ids:
            raise ValueError("Gold-type oracle could not resolve every question type")
    system_prompt_template = _prompt(prompt_style)
    run_config = {
        "family": family,
        "arm": arm,
        "stage": "sealed_context_generation",
        "population": population,
        "rows": len(contexts),
        "model": model,
        "workers": workers,
        "provider": urlparse(config.openai_base_url).netloc,
        "contexts_sha256": sha256(contexts_path),
        "prompt_style": prompt_style,
        "system_prompt_hash": stable_hash(system_prompt_template),
        "response_format": GatewayClient.response_format,
        "pre_inference_gold_access": "question_type_only"
        if prompt_style == "gold_type_oracle"
        else "none",
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": (
            "feasibility_only"
            if limit is not None
            else "diagnostic_oracle"
            if prompt_style == "gold_type_oracle"
            else "candidate_evaluation"
        ),
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"{family}-{arm}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    inference_path, scored_path = output_dir / "inference.jsonl", output_dir / "scored.jsonl"
    summary_path = output_dir / "summary.json"
    client = GatewayClient()

    def generate(row: dict[str, Any]) -> dict[str, Any]:
        if row.get("failed"):
            return {
                "question_id": str(row["question_id"]),
                "answer": None,
                "context_hash": row["context_hash"],
                "evidence_hash": row["context_hash"],
                "evidence_set_hash": row["evidence_set_hash"],
                "failed": True,
                "error_type": "UpstreamContextFailure",
            }
        system_prompt = _prompt(prompt_style, gold_types.get(str(row["question_id"])))
        context_text = "" if prompt_style == "closed_book" else str(row["context"])
        effective_context_hash = stable_hash(context_text)
        user_prompt = answer_prompt(str(row["question"]), context_text)
        try:
            result = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                max_output_tokens=800,
            )
            allowed_pmids = set(PMID_LABEL.findall(context_text))
            citations = result.answer.citation_pmids
            return {
                "question_id": str(row["question_id"]),
                "answer": result.answer.model_dump(),
                "context_hash": effective_context_hash,
                "evidence_hash": effective_context_hash,
                "evidence_set_hash": row["evidence_set_hash"],
                "prompt_hash": prompt_hash(system_prompt, user_prompt),
                "citation_count": len(citations),
                "valid_citation_count": sum(pmid in allowed_pmids for pmid in citations),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": result.latency_ms,
                "attempts": result.attempts,
                "resolved_model": result.model,
                "raw_response": result.raw_response,
                "cached": result.cached,
                "failed": False,
            }
        except Exception as exc:
            return {
                "question_id": str(row["question_id"]),
                "answer": None,
                "context_hash": effective_context_hash,
                "evidence_hash": effective_context_hash,
                "evidence_set_hash": row["evidence_set_hash"],
                "failed": True,
                "error_type": type(exc).__name__,
            }

    inference: list[dict[str, Any]] = []
    with tracked_run(run_name, run_config):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(generate, row): row for row in contexts}
            for future in as_completed(futures):
                inference.append(future.result())
        inference.sort(key=lambda row: str(row["question_id"]))
        _write_jsonl(inference_path, inference)

        # Scoring boundary: gold fields are opened only after the inference file is sealed.
        by_id = {
            str(row["question_id"]): row
            for row in iter_jsonl(_question_source(population))
            if str(row["question_id"]) in {item["question_id"] for item in inference}
        }
        scored = []
        for item in inference:
            gold = by_id[item["question_id"]]
            answer = item["answer"]
            ideal = str(answer["ideal_answer"]) if answer else ""
            scored.append(
                item
                | {
                    "question_type": str(gold["type"]),
                    "rouge_su4": rouge_su4(ideal, str(gold["answer"])),
                    "type_correct": bool(answer and answer["predicted_type"] == gold["type"]),
                }
            )
        rouge2_values = rouge2(
            [str(item["answer"]["ideal_answer"]) if item["answer"] else "" for item in scored],
            [str(by_id[item["question_id"]]["answer"]) for item in scored],
        )
        for item, value in zip(scored, rouge2_values, strict=True):
            item["rouge2_f1"] = value
        _write_jsonl(scored_path, scored)
        if population == "heldout340":
            from medrag_lab.experiments.final import record_heldout_access

            record_heldout_access(f"{family}.{arm}.scoring", scored_path)
        failures = sum(row["failed"] for row in scored)
        successful = [row for row in scored if not row["failed"]]
        latencies = [float(row["latency_ms"]) for row in successful]
        total_citations = sum(int(row["citation_count"]) for row in successful)
        valid_citations = sum(int(row["valid_citation_count"]) for row in successful)
        metrics = {
            "rouge_su4_f1": statistics.fmean(row["rouge_su4"]["f1"] for row in scored),
            "rouge2_f1": statistics.fmean(float(row["rouge2_f1"]) for row in scored),
            "question_type_accuracy": statistics.fmean(
                float(row["type_correct"]) for row in scored
            ),
            "citation_validity": valid_citations / total_citations if total_citations else 1.0,
            "citation_coverage": statistics.fmean(
                float(int(row["citation_count"]) > 0) for row in successful
            )
            if successful
            else 0.0,
            "abstention_rate": statistics.fmean(
                float(row["answer"]["abstained"]) for row in successful
            )
            if successful
            else 0.0,
            "retry_rate": statistics.fmean(float(int(row["attempts"]) > 1) for row in successful)
            if successful
            else 0.0,
            "questions": len(scored),
            "failures": failures,
            "failure_rate": failures / len(scored) if scored else 0.0,
            "input_tokens": sum(int(row["input_tokens"]) for row in successful),
            "output_tokens": sum(int(row["output_tokens"]) for row in successful),
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
            "latency_ms_p95": nearest_rank_percentile(latencies, 0.95) if latencies else 0.0,
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data_real_gateway",
            "config": run_config,
            "metrics": metrics,
            "resolved_models": sorted({str(row["resolved_model"]) for row in successful}),
            "pricing_status": (
                "USD cost unavailable from gateway; token and latency totals reported"
            ),
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
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_artifact(inference_path)
        log_artifact(scored_path)
        log_artifact(summary_path)
    return summary


def score_bertscore_artifact(
    scored_path: Path,
    population: str,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Run expensive BERTScore only on frozen finalist artifacts."""
    _guard_heldout(population)
    rows = list(iter_jsonl(scored_path))
    identifiers = {str(row["question_id"]) for row in rows}
    gold = {
        str(row["question_id"]): str(row["answer"])
        for row in iter_jsonl(_question_source(population))
        if str(row["question_id"]) in identifiers
    }
    if set(gold) != identifiers:
        raise ValueError("BERTScore could not resolve all references")
    predictions = [str(row["answer"]["ideal_answer"]) if row.get("answer") else "" for row in rows]
    references = [gold[str(row["question_id"])] for row in rows]
    values = bertscore(predictions, references, device=device)
    scored = [
        {"question_id": str(row["question_id"]), "bertscore_f1": value}
        for row, value in zip(rows, values, strict=True)
    ]
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_semantic_metric",
        "population": population,
        "questions": len(rows),
        "source_sha256": sha256(scored_path),
        "model": "microsoft/deberta-xlarge-mnli",
        "bertscore_f1": statistics.fmean(values) if values else 0.0,
        "device": device or "auto",
    }
    result["analysis_hash"] = stable_hash(result)
    output_dir = ROOT / "artifacts" / "semantic" / result["analysis_hash"][:12]
    predictions_path = output_dir / "bertscore.jsonl"
    _write_jsonl(predictions_path, scored)
    if population == "heldout340":
        from medrag_lab.experiments.final import record_heldout_access

        record_heldout_access("bertscore", predictions_path)
    result["predictions"] = str(predictions_path.relative_to(ROOT))
    destination = ROOT / "reports" / "semantic" / f"{result['analysis_hash'][:12]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
