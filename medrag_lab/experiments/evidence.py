from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
from rank_bm25 import BM25Okapi

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import sha256, stable_hash
from medrag_lab.evaluation.bioasq import snippet_span_f1
from medrag_lab.evidence.chunking import fixed_token_chunks
from medrag_lab.evidence.snippets import (
    Snippet,
    document_snippet_candidates,
    rank_snippets_cross_encoder_many,
)
from medrag_lab.experiments.runner import _write_jsonl, git_sha
from medrag_lab.indexing.bm25 import tokenize
from medrag_lab.schemas import RetrievedDocument
from medrag_lab.settings import ROOT, settings
from medrag_lab.tracking.mlflow_tracking import log_artifact, tracked_run

EvidenceArm = Literal[
    "full_document_fields",
    "fixed256_bm25",
    "sentence3_bm25",
    "sentence3_cross_encoder",
]


def _title_snippet(document: RetrievedDocument) -> Snippet:
    return Snippet(
        pmid=document.pmid,
        title=document.title,
        text=document.title,
        score=document.score,
        url=document.url,
        section="title",
        begin=0,
        end=len(document.title),
    )


def _abstract_snippet(document: RetrievedDocument) -> Snippet:
    return Snippet(
        pmid=document.pmid,
        title=document.title,
        text=document.text,
        score=document.score,
        url=document.url,
        section="abstract",
        begin=0,
        end=len(document.text),
    )


def _rank_bm25(question: str, candidates: list[Snippet], limit: int) -> list[Snippet]:
    model = BM25Okapi([tokenize(item.text) or ["__empty__"] for item in candidates])
    scores = model.get_scores(tokenize(question))
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (-float(scores[pair[0]]), -pair[1].score, pair[0]),
    )[:limit]
    return [Snippet(**(vars(item) | {"score": float(scores[index])})) for index, item in ranked]


def _annotation(snippet: Snippet) -> dict[str, Any]:
    return {
        "document": f"http://www.ncbi.nlm.nih.gov/pubmed/{snippet.pmid}",
        "beginSection": snippet.section,
        "endSection": snippet.section,
        "offsetInBeginSection": snippet.begin,
        "offsetInEndSection": snippet.end,
        "text": snippet.text,
    }


def run_evidence_retrieval(
    arm: EvidenceArm,
    retrieval_predictions: Path,
    population: str = "selection4849",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Evaluate E04 with the exact same ranked PMIDs for every evidence arm."""
    if arm not in {
        "full_document_fields",
        "fixed256_bm25",
        "sentence3_bm25",
        "sentence3_cross_encoder",
    }:
        raise ValueError(f"Unsupported evidence arm: {arm}")
    config = settings()
    if population == "heldout340":
        from medrag_lab.experiments.final import verify_final_freeze

        verify_final_freeze()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    retrieval = {
        str(row["question_id"]): row
        for row in iter_jsonl(retrieval_predictions)
        if str(row["question_id"]) in allowed
    }
    question_source = config.medrag_data_dir / (
        "eval.jsonl" if population == "heldout340" else "dev.jsonl"
    )
    questions = {
        str(row["question_id"]): {"question": str(row["question"])}
        for row in iter_jsonl(question_source)
        if str(row["question_id"]) in allowed
    }
    identifiers = sorted(set(retrieval) & set(questions))
    if set(identifiers) != allowed:
        raise ValueError("Retrieval predictions do not cover the requested population exactly")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    identifiers = identifiers[offset : offset + limit if limit is not None else None]
    corpus = {str(row["id"]): row for row in iter_jsonl(config.medrag_data_dir / "corpus.jsonl")}
    reranker = None
    if arm == "sentence3_cross_encoder":
        from medrag_lab.retrieval.reranker import MedCPTReranker

        reranker = MedCPTReranker()
    run_config = {
        "family": "E04",
        "arm": arm,
        "population": population,
        "rows": len(identifiers),
        "offset": offset,
        "retrieval_predictions_hash": stable_hash(
            {question_id: retrieval[question_id]["ranked_pmids"] for question_id in identifiers}
        ),
        "document_count": 10,
        "snippet_count": 20,
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "candidate_shard" if limit is not None or offset else "candidate_evaluation",
    }
    if reranker:
        run_config["cross_encoder_revision"] = reranker.revision
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E04-{arm}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    inference_path = output_dir / "inference_predictions.jsonl"
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    cross_ranked: dict[str, tuple[list[Snippet], float]] = {}
    if arm == "sentence3_cross_encoder" and reranker:
        prepared: list[tuple[str, list[Snippet]]] = []
        prepared_ids: list[str] = []
        for question_id in identifiers:
            ranked_pmids = list(map(str, retrieval[question_id]["ranked_pmids"][:10]))
            documents = [
                RetrievedDocument(
                    pmid=pmid,
                    title=str(corpus[pmid].get("title", "")),
                    text=str(corpus[pmid].get("text", "")),
                    url=str(
                        corpus[pmid].get("url")
                        or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    ),
                    score=float(10 - rank),
                    rank=rank,
                    retriever="frozen_e04_input",
                )
                for rank, pmid in enumerate(ranked_pmids, 1)
                if pmid in corpus
            ]
            prepared_ids.append(question_id)
            prepared.append(
                (str(questions[question_id]["question"]), document_snippet_candidates(documents))
            )
        for start in range(0, len(prepared), 16):
            batch = prepared[start : start + 16]
            ranked_batch = rank_snippets_cross_encoder_many(
                batch, reranker, 20, batch_size=64
            )
            for question_id, result in zip(
                prepared_ids[start : start + 16], ranked_batch, strict=True
            ):
                cross_ranked[question_id] = result
    with tracked_run(run_name, run_config):
        for question_id in identifiers:
            row = questions[question_id]
            try:
                ranked_pmids = list(map(str, retrieval[question_id]["ranked_pmids"][:10]))
                documents = [
                    RetrievedDocument(
                        pmid=pmid,
                        title=str(corpus[pmid].get("title", "")),
                        text=str(corpus[pmid].get("text", "")),
                        url=str(
                            corpus[pmid].get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        ),
                        score=float(10 - rank),
                        rank=rank,
                        retriever="frozen_e04_input",
                    )
                    for rank, pmid in enumerate(ranked_pmids, 1)
                    if pmid in corpus
                ]
                title_candidates = [_title_snippet(document) for document in documents]
                if arm == "full_document_fields":
                    selected = [
                        snippet
                        for document in documents
                        for snippet in (_title_snippet(document), _abstract_snippet(document))
                    ][:20]
                    latency = 0.0
                elif arm == "fixed256_bm25":
                    selected = _rank_bm25(
                        str(row["question"]), title_candidates + fixed_token_chunks(documents), 20
                    )
                    latency = 0.0
                else:
                    candidates = document_snippet_candidates(documents)
                    if arm == "sentence3_bm25":
                        selected = _rank_bm25(str(row["question"]), candidates, 20)
                        latency = 0.0
                    else:
                        selected, latency = cross_ranked[question_id]
                predicted = [_annotation(item) for item in selected]
                latencies.append(latency)
                rows.append(
                    {
                        "question_id": question_id,
                        "ranked_pmids_hash": stable_hash(ranked_pmids),
                        "snippets": predicted,
                        "latency_ms": latency,
                        "failed": False,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "question_id": question_id,
                        "snippets": [],
                        "latency_ms": 0.0,
                        "failed": True,
                        "error_type": type(exc).__name__,
                    }
                )
        _write_jsonl(inference_path, rows)
        if population == "heldout340":
            from medrag_lab.experiments.final import record_heldout_access

            record_heldout_access("E11.evidence.inference", inference_path)
        gold = {
            str(row["question_id"]): row
            for row in iter_jsonl(question_source)
            if str(row["question_id"]) in {item["question_id"] for item in rows}
        }
        for item in rows:
            reference = gold[item["question_id"]]
            metrics = snippet_span_f1(item["snippets"], reference["snippets"])
            gold_pmids = {
                str(value["document"]).rstrip("/").rsplit("/", 1)[-1]
                for value in reference["snippets"]
            }
            predicted_pmids = {
                str(value["document"]).rstrip("/").rsplit("/", 1)[-1]
                for value in item["snippets"]
            }
            metrics["gold_pmid_recall"] = (
                len(gold_pmids & predicted_pmids) / len(gold_pmids) if gold_pmids else 0.0
            )
            item["question_type"] = str(reference["type"])
            item["metrics"] = metrics
        _write_jsonl(predictions_path, rows)
        if population == "heldout340":
            record_heldout_access("E11.evidence.scoring", predictions_path)
        aggregate = {
            f"snippet_span_{name}": statistics.fmean(item["metrics"][name] for item in rows)
            for name in ("precision", "recall", "f1")
        }
        aggregate["gold_pmid_recall"] = statistics.fmean(
            item["metrics"]["gold_pmid_recall"] for item in rows
        )
        failures = sum(item["failed"] for item in rows)
        aggregate |= {
            "questions": len(rows),
            "failures": failures,
            "failure_rate": failures / len(rows) if rows else 0.0,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
        }
        summary = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "observed_real_data",
            "scope": "gold snippet offsets are used only after evidence selection",
            "config": run_config,
            "metrics": aggregate,
            "artifacts": {
                "inference_predictions": str(inference_path.relative_to(ROOT)),
                "predictions": str(predictions_path.relative_to(ROOT)),
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report_path = ROOT / "reports" / "runs" / f"{run_name}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        mlflow.log_metrics({key: float(value) for key, value in aggregate.items()})
        log_artifact(inference_path)
        log_artifact(predictions_path)
        log_artifact(summary_path)
    return summary


def merge_evidence_shards(
    source_paths: list[Path], population: str, arm: EvidenceArm
) -> dict[str, Any]:
    """Merge bounded E04 cross-encoder shards after verifying exact population coverage."""
    if not source_paths:
        raise ValueError("At least one evidence shard is required")
    config = settings()
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    allowed = set(map(str, splits[population]))
    selected: dict[str, dict[str, Any]] = {}
    for path in source_paths:
        for row in iter_jsonl(path):
            question_id = str(row["question_id"])
            if question_id not in allowed:
                continue
            previous = selected.get(question_id)
            if previous is None or (previous.get("failed") and not row.get("failed")):
                selected[question_id] = row
            elif (
                not previous.get("failed")
                and not row.get("failed")
                and previous.get("snippets") != row.get("snippets")
            ):
                raise ValueError(f"Conflicting evidence predictions for {question_id}")
    if set(selected) != allowed:
        raise ValueError(f"Evidence shards are missing {len(allowed - set(selected))} question IDs")
    rows = [selected[question_id] for question_id in sorted(selected)]
    failures = sum(bool(row.get("failed")) for row in rows)
    if failures:
        raise ValueError(f"Merged evidence still contains {failures} failures")
    run_config = {
        "family": "E04",
        "arm": arm,
        "population": population,
        "rows": len(rows),
        "merge_policy": "successful_retry_replaces_failed_attempt",
        "source_sha256": [sha256(path) for path in source_paths],
        "split_freeze_hash": splits["freeze_hash"],
        "git_sha": git_sha(),
        "purpose": "candidate_evaluation",
    }
    run_config["config_hash"] = stable_hash(run_config)
    run_name = f"E04-{arm}-{population}-{run_config['config_hash'][:10]}"
    output_dir = config.medrag_artifact_dir / run_name
    predictions_path, summary_path = output_dir / "predictions.jsonl", output_dir / "summary.json"
    _write_jsonl(predictions_path, rows)
    metrics = {
        f"snippet_span_{name}": statistics.fmean(float(row["metrics"][name]) for row in rows)
        for name in ("precision", "recall", "f1")
    }
    metrics |= {
        "gold_pmid_recall": statistics.fmean(
            float(row["metrics"]["gold_pmid_recall"]) for row in rows
        ),
        "questions": len(rows),
        "failures": 0,
        "failure_rate": 0.0,
        "latency_ms_p50": statistics.median(float(row["latency_ms"]) for row in rows),
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_real_data_merged_shards",
        "scope": "gold snippet offsets are used only after evidence selection",
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
