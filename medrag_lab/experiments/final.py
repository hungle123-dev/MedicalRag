from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from medrag_lab.data.manifests import atomic_json, sha256, stable_hash
from medrag_lab.evaluation.statistics import holm_adjust
from medrag_lab.pipeline import load_pipeline_config
from medrag_lab.settings import ROOT

FINAL_PIPELINES = (
    "closed_book",
    "bm25_rag",
    "best_rag",
    "preselected_drop_one",
    "gold_evidence_oracle",
)
FINAL_CONTRASTS = (
    "best_vs_vanilla_bm25",
    "best_vs_closed_book",
    "best_vs_preselected_drop_one",
)


def freeze_finalists(
    pipeline_ids: tuple[str, ...] = FINAL_PIPELINES,
    destination: Path | None = None,
) -> dict:
    if tuple(pipeline_ids) != FINAL_PIPELINES:
        raise ValueError("Final freeze requires exactly the preregistered five arms in order")
    configs = {pipeline_id: load_pipeline_config(pipeline_id) for pipeline_id in pipeline_ids}
    inventory = ROOT / "artifacts" / "gateway" / "model_inventory.json"
    if not inventory.is_file():
        raise FileNotFoundError("Run gateway model preflight before freezing finalists")
    payload = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "pipeline_ids": list(pipeline_ids),
        "pipeline_config_hashes": {
            pipeline_id: stable_hash(value) for pipeline_id, value in configs.items()
        },
        "contrasts": list(FINAL_CONTRASTS),
        "holm_family_size": 3,
        "judge_config_sha256": sha256(ROOT / "configs" / "judges" / "panel.yaml"),
        "model_inventory_sha256": sha256(inventory),
        "heldout_seen": False,
    }
    payload["freeze_hash"] = stable_hash(payload)
    path = destination or ROOT / "data" / "manifests" / "final_freeze.json"
    atomic_json(path, payload)
    return payload


def verify_final_freeze(
    path: Path | None = None, *, require_source_unchanged: bool = True
) -> dict:
    source = path or ROOT / "data" / "manifests" / "final_freeze.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    freeze_hash = payload.pop("freeze_hash")
    if stable_hash(payload) != freeze_hash:
        raise ValueError("Final freeze manifest hash mismatch")
    for pipeline_id, expected in payload["pipeline_config_hashes"].items():
        if stable_hash(load_pipeline_config(pipeline_id)) != expected:
            raise ValueError(f"Frozen pipeline changed: {pipeline_id}")
    source_unchanged = subprocess.run(
        ["git", "diff", "--quiet", payload["git_sha"], "--", "medrag_lab", "apps", "configs"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    if require_source_unchanged and not source_unchanged:
        raise ValueError("Source changed since final freeze; held-out rerun is blocked")
    return {
        "status": "verified_historical_manifest",
        "freeze_hash": freeze_hash,
        "source_unchanged": source_unchanged,
        "heldout_rerun_allowed": source_unchanged,
    }


def apply_final_holm(comparison_paths: list[Path]) -> dict:
    if len(comparison_paths) != 3:
        raise ValueError("Holm correction is locked to exactly three preregistered contrasts")
    comparisons = [json.loads(path.read_text(encoding="utf-8")) for path in comparison_paths]
    raw = [float(value["paired_permutation_p"]) for value in comparisons]
    adjusted = holm_adjust(raw)
    result = {
        "family_size": 3,
        "contrasts": [
            value | {"holm_adjusted_p": corrected}
            for value, corrected in zip(comparisons, adjusted, strict=True)
        ],
    }
    result["analysis_hash"] = stable_hash(result)
    destination = ROOT / "reports" / "final_statistics.json"
    atomic_json(destination, result)
    return result


def record_heldout_access(stage: str, artifact: Path) -> dict:
    """Append-only audit ledger written only after a sealed held-out artifact exists."""
    freeze = verify_final_freeze()
    destination = ROOT / "reports" / "heldout_access.json"
    ledger = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else {}
    events = list(ledger.get("events", []))
    event = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_sha256": sha256(artifact),
        "freeze_hash": freeze["freeze_hash"],
    }
    if not any(
        row.get("stage") == stage and row.get("artifact_sha256") == event["artifact_sha256"]
        for row in events
    ):
        events.append(event)
    payload = {"events": events}
    payload["ledger_hash"] = stable_hash(payload)
    atomic_json(destination, payload)
    return payload
