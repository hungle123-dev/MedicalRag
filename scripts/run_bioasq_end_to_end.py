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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.env import load_dotenv
from app.pipelines import PIPELINES, build_e5_arms, generate_from_evidence

ARMS = ("B3", "G2", "X1", "X2")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float]) -> float:
    ordered = sorted(values); middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            text=True, capture_output=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
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
    model_slug = "".join(character if character.isalnum() else "-" for character in model).strip("-")
    run_id = f"bioasq_{args.split}_e5_{generator}_{model_slug}_v7_{count}_{commit[:8]}"
    directory = ROOT / "artifacts/experiments/bioasq" / run_id
    directory.mkdir(parents=True, exist_ok=True)

    dev_warmup = json.loads((ROOT / "data/raw/bioasq/dev.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    PIPELINES["B3"].run(dev_warmup["question"])
    started = datetime.now(timezone.utc).isoformat()
    records, snapshot_hashes = [], []
    for question_index, question_id in enumerate(ids):
        row = by_id[question_id]
        snapshot_target = directory / f"{question_id}_evidence.json"
        arms = build_e5_arms(row["question"], seed=frozen["seed"] + question_index)
        snapshot = {"question_id": question_id, "arms": arms,
                    "experiment_config_hash": sha(ROOT / "configs/experiments.yaml")}
        snapshot_text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_text.encode()).hexdigest()
        snapshot_hashes.append(snapshot_hash)
        snapshot_target.write_text(json.dumps(snapshot | {"snapshot_hash": snapshot_hash},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
        offset = question_index % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        for arm_id in order:
            target = directory / f"{question_id}_{arm_id}.json"
            if target.exists() and not args.force:
                record = json.loads(target.read_text(encoding="utf-8"))
            else:
                try:
                    result = generate_from_evidence(row["question"], arm_id, arms[arm_id])
                    record = {"status": "completed", "question_id": question_id, "pipeline_id": arm_id,
                              "question": row["question"], "reference_answer": row["answer"],
                              "evidence_snapshot_hash": snapshot_hash, "result": result}
                except Exception as exc:
                    record = {"status": "failed", "question_id": question_id, "pipeline_id": arm_id,
                              "question": row["question"], "reference_answer": row["answer"],
                              "evidence_snapshot_hash": snapshot_hash,
                              "error": f"{type(exc).__name__}"}
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, target)
            records.append(record)
    manifest = {
        "run_id": run_id, "track": "bioasq_e5_answer_stage", "split": args.split,
        "locked_execution": args.split == "eval", "arms": list(ARMS), "question_ids": ids,
        "excluded_calibration_ids": frozen["ids"][:excluded], "seed": frozen["seed"],
        "generator": generator, "generator_model": model, "code_commit": commit,
        "working_tree_dirty_at_start": dirty, "order": "latin_rotation_by_question_index",
        "warmup": "one unmeasured out-of-population dev B3 question",
        "config_hashes": {"experiments": sha(ROOT / "configs/experiments.yaml"),
                          "pipelines": sha(ROOT / "configs/pipelines.yaml"),
                          "protocol": sha(ROOT / "configs/protocol.yaml"),
                          "prompt": sha(ROOT / "configs/prompts/answer_v1.txt"),
                          "data": sha(ROOT / "data/manifests/files.json"),
                          "population": sha(id_manifest)},
        "evidence_population_hash": hashlib.sha256("".join(snapshot_hashes).encode()).hexdigest(),
        "started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records), "failures": sum(record["status"] == "failed" for record in records),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed = [record for record in records if record["status"] == "completed"]
    summary = {"run_id": run_id, "split": args.split, "questions": len(ids), "arms": list(ARMS),
               "generator_model": model, "code_commit": commit,
               "graph_positive_questions": sum(arms["G2"]["graph_positive"] for arms in (
                   json.loads((directory / f"{question_id}_evidence.json").read_text(encoding="utf-8"))["arms"]
                   for question_id in ids)),
               "failure_rate": round((len(records) - len(completed)) / len(records), 6),
               "mean_latency_ms": {arm: round(sum(record["result"]["details"]["latency_ms"] for record in completed
                                                    if record["pipeline_id"] == arm) / max(1, sum(
                                                        record["pipeline_id"] == arm for record in completed)), 3)
                                   for arm in ARMS},
               "citation_integrity_rate": {arm: round(sum(record["result"]["details"]["citation_integrity"]["valid"]
                                                           for record in completed if record["pipeline_id"] == arm) /
                                                          max(1, sum(record["pipeline_id"] == arm for record in completed)), 6)
                                           for arm in ARMS},
               "interpretation": "Machine outputs require blinded judging and qualified human review."}
    output = ROOT / f"data/manifests/{run_id}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
