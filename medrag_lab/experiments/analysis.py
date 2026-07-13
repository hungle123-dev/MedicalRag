from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import stable_hash
from medrag_lab.data.splits import normalize_question
from medrag_lab.evaluation.statistics import paired_group_bootstrap
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
