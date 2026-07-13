from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import stable_hash
from medrag_lab.data.splits import normalize_question
from medrag_lab.evaluation.statistics import nearest_rank_percentile, paired_group_bootstrap
from medrag_lab.settings import ROOT, settings


def _metric(row: dict[str, Any], path: str) -> float:
    value: Any = row
    for key in path.split("."):
        value = value[key]
    return float(value)


def analyze_two_by_two_interaction(
    a0b0_path: Path,
    a0b1_path: Path,
    a1b0_path: Path,
    a1b1_path: Path,
    metric: str,
    interaction_id: str,
) -> dict[str, Any]:
    """Estimate a paired difference-in-differences with group-aware bootstrap CI."""
    paths = (a0b0_path, a0b1_path, a1b0_path, a1b1_path)
    arms = [{str(row["question_id"]): row for row in iter_jsonl(path)} for path in paths]
    identifiers = sorted(arms[0])
    if any(set(arm) != set(identifiers) for arm in arms[1:]):
        raise ValueError("Interaction arms require identical question IDs")
    normalized: dict[str, str] = {}
    for filename in ("dev.jsonl", "eval.jsonl"):
        for row in iter_jsonl(settings().medrag_data_dir / filename):
            question_id = str(row["question_id"])
            if question_id in arms[0]:
                normalized[question_id] = normalize_question(str(row["question"]))
    if set(normalized) != set(identifiers):
        raise ValueError("Could not resolve every normalized question group")
    values = [[_metric(arm[item], metric) for item in identifiers] for arm in arms]
    did = [(a1b1 - a1b0) - (a0b1 - a0b0) for a0b0, a0b1, a1b0, a1b1 in zip(*values, strict=True)]
    bootstrap = paired_group_bootstrap(
        [0.0] * len(did), did, [normalized[item] for item in identifiers]
    )
    result = {
        "interaction_id": interaction_id,
        "arms": [str(path) for path in paths],
        "metric": metric,
        "rows": len(identifiers),
        "difference_in_differences": bootstrap["mean_delta_right_minus_left"],
        "ci95_low": bootstrap["ci95_low"],
        "ci95_high": bootstrap["ci95_high"],
        "material_interaction": bootstrap["ci95_low"] > 0 or bootstrap["ci95_high"] < 0,
    }
    result["analysis_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "interactions" / f"{interaction_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_query_strategy_gate(
    baseline_path: Path,
    candidate_path: Path,
    comparison_path: Path,
    gate_id: str,
) -> dict[str, Any]:
    baseline = {str(row["question_id"]): row for row in iter_jsonl(baseline_path)}
    candidate = {str(row["question_id"]): row for row in iter_jsonl(candidate_path)}
    if set(baseline) != set(candidate):
        raise ValueError("Query gate requires identical question IDs")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    identifiers = sorted(baseline)
    severe = sum(
        (baseline[item]["metrics"]["hit"] == 1 and candidate[item]["metrics"]["hit"] == 0)
        or baseline[item]["metrics"]["ap"] - candidate[item]["metrics"]["ap"] >= 0.20
        for item in identifiers
    )
    rescued = sum(
        baseline[item]["metrics"]["hit"] == 0 and candidate[item]["metrics"]["hit"] == 1
        for item in identifiers
    )
    baseline_failures = sum(bool(baseline[item].get("failed")) for item in identifiers)
    candidate_failures = sum(bool(candidate[item].get("failed")) for item in identifiers)
    left_latency = [float(baseline[item].get("latency_ms", 0.0)) for item in identifiers]
    right_latency = [float(candidate[item].get("latency_ms", 0.0)) for item in identifiers]
    latency_ratio = nearest_rank_percentile(right_latency, 0.95) / max(
        nearest_rank_percentile(left_latency, 0.95), 1e-9
    )
    bootstrap = comparison["bootstrap"]
    checks = {
        "minimum_effect": bootstrap["mean_delta_right_minus_left"] >= 0.01,
        "positive_paired_ci": bootstrap["ci95_low"] > 0,
        "severe_harm_below_5_percent": severe / len(identifiers) < 0.05,
        "failure_guard": (candidate_failures - baseline_failures) / len(identifiers) <= 0.005,
        "latency_guard": latency_ratio <= 2.0,
    }
    result = {
        "gate_id": gate_id,
        "rows": len(identifiers),
        "comparison_hash": comparison["comparison_hash"],
        "rescue_rate": rescued / len(identifiers),
        "severe_harm_rate": severe / len(identifiers),
        "p95_latency_ratio": latency_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["gate_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "gates" / f"{gate_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify_context_invariant(
    left_path: Path,
    right_path: Path,
    key: str,
    invariant_id: str,
) -> dict[str, Any]:
    if key not in {"candidate_evidence_hash", "evidence_set_hash", "context_hash"}:
        raise ValueError("Unsupported context invariant key")
    left = {str(row["question_id"]): row for row in iter_jsonl(left_path)}
    right = {str(row["question_id"]): row for row in iter_jsonl(right_path)}
    if set(left) != set(right):
        raise ValueError("Context invariant requires identical question IDs")
    mismatches = [item for item in sorted(left) if left[item].get(key) != right[item].get(key)]
    result = {
        "invariant_id": invariant_id,
        "key": key,
        "rows": len(left),
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "passed": not mismatches,
    }
    result["invariant_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "invariants" / f"{invariant_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def audit_failure_sequence(predictions_path: Path, incident_id: str) -> dict[str, Any]:
    rows = list(iter_jsonl(predictions_path))
    failed_indexes = [index for index, row in enumerate(rows) if row.get("failed")]
    error_types = Counter(
        str(row.get("error_type", "Unknown")) for row in rows if row.get("failed")
    )
    first_failure = min(failed_indexes) if failed_indexes else None
    successes_after_first = (
        sum(not row.get("failed") for row in rows[first_failure:])
        if first_failure is not None
        else 0
    )
    result = {
        "incident_id": incident_id,
        "predictions": str(predictions_path),
        "rows": len(rows),
        "failures": len(failed_indexes),
        "failure_rate": len(failed_indexes) / len(rows) if rows else 0.0,
        "error_types": dict(error_types),
        "first_failure_index_zero_based": first_failure,
        "successes_after_first_failure": successes_after_first,
        "contiguous_terminal_failure": bool(failed_indexes) and successes_after_first == 0,
        "interpretation": (
            "A contiguous AcceleratorError tail is consistent with process-level device/context "
            "loss, not independent per-question model errors; recover via bounded process shards."
            if set(error_types) == {"AcceleratorError"} and successes_after_first == 0
            else "Failure pattern requires case-level inspection."
        ),
    }
    result["incident_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "incidents" / f"{incident_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
