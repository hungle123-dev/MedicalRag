"""Resumable four-arm BioASQ E5 run with a frozen evidence snapshot per question."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.env import load_dotenv
from app.pipelines import (PIPELINES, bm25_index_path, build_e5_arms, generate_from_evidence,
                           graph_index_path, medcpt_index_path)

ARMS = ("B3", "G2", "X1", "X2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float:
    ordered = sorted(values); middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = round((len(ordered) - 1) * probability)
    return ordered[position]


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def validate_arm_design(arms: dict[str, dict]) -> dict:
    totals = {arm: sum(len(item.get("snippet", "").split())
                       for item in arms[arm]["evidence"]) for arm in ARMS}
    errors = []
    if len(set(totals.values())) != 1:
        errors.append(f"unequal_total_words:{totals}")
    positive = bool(arms["G2"].get("graph_retrieval_positive"))
    if positive:
        for control in ("X1", "X2"):
            if not arms[control].get("control_complete"):
                errors.append(f"{control.lower()}_incomplete:{arms[control].get('control_audit')}")
    else:
        frozen = json.dumps(arms["B3"]["evidence"], sort_keys=True)
        if any(json.dumps(arms[arm]["evidence"], sort_keys=True) != frozen for arm in ARMS[1:]):
            errors.append("graph_negative_arms_not_identical")
    if errors:
        raise RuntimeError("Invalid E5 arm design: " + "; ".join(errors))
    return {"total_words": totals["B3"], "graph_retrieval_positive": positive,
            "x1_complete": arms["X1"].get("control_complete", True),
            "x2_complete": arms["X2"].get("control_complete", True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--questions", type=int)
    parser.add_argument("--exclude-first", type=int)
    parser.add_argument("--confirm-locked", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.split == "eval" and not args.confirm_locked:
        raise SystemExit("Locked eval requires --confirm-locked after the protocol is frozen")
    if args.split == "eval" and (args.questions is not None or args.exclude_first is not None
                                 or args.force or args.allow_dirty):
        raise SystemExit("Locked eval forbids population overrides, force, and dirty-worktree execution")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                 check=True, text=True, capture_output=True).stdout)
    if dirty and not args.allow_dirty:
        raise SystemExit("Refusing a research run from a dirty tracked worktree; commit first")
    load_dotenv(ROOT)
    id_manifest = ROOT / f"data/manifests/bioasq_{'dev' if args.split == 'dev' else 'eval'}_question_ids.json"
    frozen = json.loads(id_manifest.read_text(encoding="utf-8"))
    excluded = args.exclude_first if args.exclude_first is not None else (20 if args.split == "dev" else 0)
    available_ids = frozen["ids"][excluded:]
    count = args.questions if args.questions is not None else (50 if args.split == "dev" else len(available_ids))
    ids = available_ids[:count]
    if len(ids) != count:
        raise SystemExit(f"Requested {count} questions but only {len(ids)} are frozen")
    with (ROOT / f"data/raw/bioasq/{args.split}.jsonl").open(encoding="utf-8") as stream:
        by_id = {row["question_id"]: row for line in stream if (row := json.loads(line)) and row["question_id"] in set(ids)}
    if set(by_id) != set(ids):
        raise SystemExit("Frozen IDs do not match the local BioASQ split")
    generator = os.getenv("MEDICAL_RAG_GENERATOR", "mock")
    model = os.getenv("GATEWAY_GENERATOR_MODEL", "local") if generator == "gateway" else generator
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    if args.split == "eval" and (generator != "gateway" or model != protocol["generator"]["model"]):
        raise SystemExit("Locked eval generator/provider/model do not match configs/protocol.yaml")
    model_slug = "".join(character if character.isalnum() else "-" for character in model).strip("-")
    run_id = f"bioasq_{args.split}_e5_{generator}_{model_slug}_v9_{count}_{commit[:8]}"
    directory = ROOT / "artifacts/experiments/bioasq" / run_id
    directory.mkdir(parents=True, exist_ok=True)

    index_files = [bm25_index_path(), medcpt_index_path() / "articles.faiss",
                   medcpt_index_path() / "metadata.jsonl", graph_index_path()]
    if not all(path.is_file() for path in index_files):
        raise RuntimeError("A frozen index is missing before the run")
    config_paths = {"experiments": ROOT / "configs/experiments.yaml",
                    "pipelines": ROOT / "configs/pipelines.yaml",
                    "protocol": ROOT / "configs/protocol.yaml",
                    "prompt": ROOT / "configs/prompts/answer_v1.txt",
                    "data": ROOT / "data/manifests/files.json", "population": id_manifest}
    config_hashes = {name: sha(path) for name, path in config_paths.items()}
    index_fingerprints = {path.name: file_fingerprint(path) for path in index_files}

    # Freeze and validate the entire evidence population before observing any
    # answer output. This prevents a late control failure from creating a
    # selectively generated population.
    prepared, snapshot_hashes = {}, []
    for question_index, question_id in enumerate(ids):
        row = by_id[question_id]
        snapshot_target = directory / f"{question_id}_evidence.json"
        arms = build_e5_arms(row["question"], seed=frozen["seed"] + question_index)
        design_audit = validate_arm_design(arms)
        snapshot = {"question_id": question_id, "question_type": row.get("type"), "arms": arms,
                    "design_audit": design_audit,
                    "experiment_config_hash": config_hashes["experiments"],
                    "frozen_input_hashes": {"configs": config_hashes,
                                            "indexes": index_fingerprints}}
        snapshot_text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_text.encode()).hexdigest()
        snapshot_hashes.append(snapshot_hash)
        if snapshot_target.exists() and not args.force:
            existing = json.loads(snapshot_target.read_text(encoding="utf-8"))
            if existing.get("snapshot_hash") != snapshot_hash:
                raise RuntimeError(f"Evidence snapshot changed while resuming {question_id}")
            arms = existing["arms"]
        else:
            snapshot_target.write_text(json.dumps(snapshot | {"snapshot_hash": snapshot_hash},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
        prepared[question_id] = {"arms": arms, "snapshot_hash": snapshot_hash}
        print(f"prepared_evidence={question_index + 1}/{len(ids)}", flush=True)

    dev_warmup = json.loads((ROOT / "data/raw/bioasq/dev.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    PIPELINES["B3"].run(dev_warmup["question"])
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for question_index, question_id in enumerate(ids):
        row = by_id[question_id]
        arms = prepared[question_id]["arms"]
        snapshot_hash = prepared[question_id]["snapshot_hash"]
        offset = question_index % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        for arm_id in order:
            target = directory / f"{question_id}_{arm_id}.json"
            if target.exists() and not args.force:
                record = json.loads(target.read_text(encoding="utf-8"))
                if record.get("evidence_snapshot_hash") != snapshot_hash:
                    raise RuntimeError(f"Answer/evidence hash mismatch while resuming {target.name}")
            else:
                try:
                    result = generate_from_evidence(row["question"], arm_id, arms[arm_id])
                    record = {"status": "completed", "question_id": question_id, "pipeline_id": arm_id,
                              "question": row["question"], "question_type": row.get("type"),
                              "reference_answer": row["answer"],
                              "evidence_snapshot_hash": snapshot_hash, "result": result}
                except Exception as exc:
                    record = {"status": "failed", "question_id": question_id, "pipeline_id": arm_id,
                              "question": row["question"], "question_type": row.get("type"),
                              "reference_answer": row["answer"],
                              "evidence_snapshot_hash": snapshot_hash,
                              "error": f"{type(exc).__name__}"}
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, target)
            records.append(record)
        completed_questions = question_index + 1
        print(f"completed_questions={completed_questions}/{len(ids)} records={len(records)}", flush=True)
    if {name: sha(path) for name, path in config_paths.items()} != config_hashes:
        raise RuntimeError("A frozen config or population file changed during the run")
    if {path.name: file_fingerprint(path) for path in index_files} != index_fingerprints:
        raise RuntimeError("A frozen index changed during the run")
    completed = [record for record in records if record["status"] == "completed"]
    response_models = sorted({record["result"]["details"]["generator"].get("response_model")
                              or record["result"]["details"]["generator"]["model"]
                              for record in completed})
    if len(response_models) != 1:
        raise RuntimeError(f"Research run observed model drift: {response_models}")
    manifest = {
        "run_id": run_id, "track": "bioasq_e5_answer_stage", "split": args.split,
        "locked_execution": args.split == "eval", "arms": list(ARMS), "question_ids": ids,
        "excluded_calibration_ids": frozen["ids"][:excluded], "seed": frozen["seed"],
        "generator": generator, "generator_model": model, "code_commit": commit,
        "working_tree_dirty_at_start": dirty, "order": "latin_rotation_by_question_index",
        "warmup": "one unmeasured out-of-population dev B3 question",
        "config_hashes": config_hashes,
        "index_fingerprints": index_fingerprints,
        "evidence_population_hash": hashlib.sha256("".join(snapshot_hashes).encode()).hexdigest(),
        "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records), "failures": sum(record["status"] == "failed" for record in records),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed_by_arm = {arm: [record for record in completed if record["pipeline_id"] == arm]
                        for arm in ARMS}
    latency_by_arm = {arm: [record["result"]["details"]["latency_ms"] for record in completed
                            if record["pipeline_id"] == arm] for arm in ARMS}
    summary = {"run_id": run_id, "split": args.split, "questions": len(ids), "arms": list(ARMS),
               "generator_model": model, "code_commit": commit,
               "observed_response_models": response_models,
               "response_model_consistent": len(response_models) == 1,
               "graph_retrieval_positive_questions": sum(
                   arms["G2"]["graph_retrieval_positive"] for arms in (
                   json.loads((directory / f"{question_id}_evidence.json").read_text(encoding="utf-8"))["arms"]
                   for question_id in ids)),
               "failure_rate": round((len(records) - len(completed)) / len(records), 6),
               "latency_ms": {arm: {
                   "mean": round(sum(latency_by_arm[arm]) / max(1, len(latency_by_arm[arm])), 3),
                   "p50": round(median(latency_by_arm[arm]), 3) if latency_by_arm[arm] else None,
                   "p95": round(percentile(latency_by_arm[arm], .95), 3) if latency_by_arm[arm] else None,
               } for arm in ARMS},
               "api_usage": {arm: {
                    "logical_requests": len(completed_by_arm[arm]),
                    "network_calls": sum(not row["result"]["details"]["generator"].get("cached", False)
                                         for row in completed_by_arm[arm]),
                    "cache_hits": sum(row["result"]["details"]["generator"].get("cached", False)
                                      for row in completed_by_arm[arm]),
                    "logical_prompt_tokens": sum(int((row["result"]["details"]["generator"].get("usage") or {}).get("prompt_tokens", 0))
                                                 for row in completed_by_arm[arm]),
                    "network_prompt_tokens": sum(int((row["result"]["details"]["generator"].get("usage") or {}).get("prompt_tokens", 0))
                                                 for row in completed_by_arm[arm]
                                                 if not row["result"]["details"]["generator"].get("cached", False)),
                    "network_completion_tokens": sum(int((row["result"]["details"]["generator"].get("usage") or {}).get("completion_tokens", 0))
                                                     for row in completed_by_arm[arm]
                                                     if not row["result"]["details"]["generator"].get("cached", False)),
                } for arm in ARMS},
               "no_invented_citation_id_rate": {arm: round(sum(record["result"]["details"]["citation_integrity"]["valid"]
                                                           for record in completed if record["pipeline_id"] == arm) /
                                                          max(1, sum(record["pipeline_id"] == arm for record in completed)), 6)
                                           for arm in ARMS},
               "interpretation": "Machine outputs require blinded judging and qualified human review."}
    output = ROOT / f"data/manifests/{run_id}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
