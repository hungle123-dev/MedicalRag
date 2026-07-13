from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import stable_hash
from medrag_lab.evaluation.llm_panel import LLMPanel
from medrag_lab.experiments.runner import _write_jsonl
from medrag_lab.settings import ROOT, settings


def _source(population: str) -> Path:
    return settings().medrag_data_dir / (
        "eval.jsonl" if population in {"heldout340", "judge160"} else "dev.jsonl"
    )


def _selected_ids(population: str, limit: int | None) -> list[str]:
    splits = json.loads((ROOT / "data" / "manifests" / "splits.json").read_text("utf-8"))
    identifiers = sorted(map(str, splits[population]))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        identifiers = identifiers[:limit]
    return identifiers


def run_panel_direct(
    generation_path: Path,
    contexts_path: Path,
    population: str,
    *,
    limit: int | None = None,
    workers: int = 2,
) -> dict[str, Any]:
    identifiers = _selected_ids(population, limit)
    selected = set(identifiers)
    generated = {
        str(row["question_id"]): row
        for row in iter_jsonl(generation_path)
        if str(row["question_id"]) in selected
    }
    contexts = {
        str(row["question_id"]): row
        for row in iter_jsonl(contexts_path)
        if str(row["question_id"]) in selected
    }
    gold = {
        str(row["question_id"]): row
        for row in iter_jsonl(_source(population))
        if str(row["question_id"]) in selected
    }
    if any(set(values) != selected for values in (generated, contexts, gold)):
        raise ValueError("Panel inputs do not cover the requested IDs exactly")
    panel = LLMPanel()

    def judge(question_id: str) -> dict[str, Any]:
        generation = generated[question_id]
        if generation.get("failed") or not generation.get("answer"):
            return {"question_id": question_id, "failed": True, "error_type": "UpstreamFailure"}
        try:
            result = panel.direct(
                str(gold[question_id]["question"]),
                str(generation["answer"]["ideal_answer"]),
                str(gold[question_id]["answer"]),
                str(contexts[question_id]["context"]),
            )
            return {"question_id": question_id, **result, "failed": False}
        except Exception as exc:
            return {
                "question_id": question_id,
                "failed": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge, question_id): question_id for question_id in identifiers}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["question_id"])
    successful = [row for row in rows if not row["failed"]]
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_real_gateway_automated_proxy",
        "population": population,
        "questions": len(rows),
        "failures": len(rows) - len(successful),
        "median_panel_score_0_4": statistics.median(
            row["median_weighted_score_0_4"] for row in successful
        )
        if successful
        else 0.0,
        "mean_unsupported_atomic_claim_rate": statistics.fmean(
            row["median_unsupported_atomic_claim_rate"] for row in successful
        )
        if successful
        else 1.0,
        "disagreement_rate": statistics.fmean(float(row["disagreement_flag"]) for row in successful)
        if successful
        else 0.0,
        "limitation": "Automated multi-LLM proxy; not human or physician review",
    }
    analysis_hash = stable_hash(result)
    result["analysis_hash"] = analysis_hash
    output_dir = ROOT / "artifacts" / "panel" / analysis_hash[:12]
    predictions_path = output_dir / "direct.jsonl"
    _write_jsonl(predictions_path, rows)
    result["predictions"] = str(predictions_path.relative_to(ROOT))
    destination = ROOT / "reports" / "panel" / f"direct-{analysis_hash[:12]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_panel_pairwise(
    left_path: Path,
    right_path: Path,
    contexts_path: Path,
    population: str = "judge160",
    *,
    limit: int | None = None,
    workers: int = 2,
) -> dict[str, Any]:
    identifiers = _selected_ids(population, limit)
    selected = set(identifiers)
    left = {
        str(row["question_id"]): row
        for row in iter_jsonl(left_path)
        if str(row["question_id"]) in selected
    }
    right = {
        str(row["question_id"]): row
        for row in iter_jsonl(right_path)
        if str(row["question_id"]) in selected
    }
    contexts = {
        str(row["question_id"]): row
        for row in iter_jsonl(contexts_path)
        if str(row["question_id"]) in selected
    }
    questions = {
        str(row["question_id"]): str(row["question"])
        for row in iter_jsonl(_source(population))
        if str(row["question_id"]) in selected
    }
    if any(set(values) != selected for values in (left, right, contexts, questions)):
        raise ValueError("Pairwise panel inputs do not cover the requested IDs exactly")
    panel = LLMPanel()

    def judge(question_id: str) -> dict[str, Any]:
        left_answer, right_answer = (
            left[question_id].get("answer"),
            right[question_id].get("answer"),
        )
        if not left_answer or not right_answer:
            return {"question_id": question_id, "failed": True, "error_type": "UpstreamFailure"}
        try:
            result = panel.pairwise(
                question_id,
                questions[question_id],
                str(left_answer["ideal_answer"]),
                str(right_answer["ideal_answer"]),
                str(contexts[question_id]["context"]),
            )
            return {"question_id": question_id, **result, "failed": False}
        except Exception as exc:
            return {
                "question_id": question_id,
                "failed": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(judge, question_id): question_id for question_id in identifiers}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["question_id"])
    successful = [row for row in rows if not row["failed"]]
    winners = {
        name: sum(row["panel_winner"] == name for row in successful)
        for name in ("left", "right", "tie")
    }
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_real_gateway_position_swapped_proxy",
        "population": population,
        "questions": len(rows),
        "failures": len(rows) - len(successful),
        "panel_winners": winners,
        "limitation": "Automated multi-LLM proxy; not human or physician review",
    }
    analysis_hash = stable_hash(result)
    result["analysis_hash"] = analysis_hash
    output_dir = ROOT / "artifacts" / "panel" / analysis_hash[:12]
    predictions_path = output_dir / "pairwise.jsonl"
    _write_jsonl(predictions_path, rows)
    result["predictions"] = str(predictions_path.relative_to(ROOT))
    destination = ROOT / "reports" / "panel" / f"pairwise-{analysis_hash[:12]}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
