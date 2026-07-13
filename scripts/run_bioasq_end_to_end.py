"""Resumable BioASQ B3/G2 run; cached per-question artifacts are replayed by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipelines import PIPELINES


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Ignore cached result files")
    args = parser.parse_args()
    frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
    ids = frozen["ids"][:args.questions]
    with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
        by_id = {row["question_id"]: row for line in stream if (row := json.loads(line))}
    generator = os.getenv("MEDICAL_RAG_GENERATOR", "mock")
    run_id = f"bioasq_dev_b3_g2_{generator}_warmed_counterbalanced_v4_{len(ids)}_20260712"
    directory = ROOT / "artifacts/experiments/bioasq" / run_id; directory.mkdir(parents=True, exist_ok=True)
    warmup_question = by_id[ids[0]]["question"]
    PIPELINES["B3"].run(warmup_question); PIPELINES["G2"].run(warmup_question)
    started = datetime.now(timezone.utc).isoformat(); results = []
    for question_index, question_id in enumerate(ids):
        order = ("B3", "G2") if question_index % 2 == 0 else ("G2", "B3")
        for pipeline_id in order:
            target = directory / f"{question_id}_{pipeline_id}.json"
            if target.exists() and not args.force:
                result = json.loads(target.read_text(encoding="utf-8"))
            else:
                row = by_id[question_id]
                result = {"question_id": question_id, "pipeline_id": pipeline_id, "question": row["question"],
                          "reference_answer": row["answer"], "result": PIPELINES[pipeline_id].run(row["question"])}
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, target)
            results.append(result)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    manifest = {"run_id": run_id, "track": "bioasq_dev_smoke", "pipelines": ["B3", "G2"],
        "generator": generator, "question_ids": ids, "seed": frozen["seed"], "code_commit": commit,
        "working_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout),
        "pipeline_order": "counterbalanced_by_question_index",
        "warmup": "one unmeasured B3 and G2 call before the paired population",
        "config_hash": sha(ROOT / "configs/pipelines.yaml"), "prompt_hash": sha(ROOT / "configs/prompts/answer_v1.txt"),
        "data_manifest_hash": sha(ROOT / "data/manifests/files.json"), "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(), "result_files": len(results),
        "mock_is_not_a_clinical_result": generator == "mock"}
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paired = {question_id: {} for question_id in ids}
    for item in results: paired[item["question_id"]][item["pipeline_id"]] = item["result"]["details"]["latency_ms"]
    summary = {"run_id": run_id, "questions": len(ids), "generator": generator,
        "mean_latency_ms": {pipeline: round(sum(item["result"]["details"]["latency_ms"] for item in results
                                              if item["pipeline_id"] == pipeline) / len(ids), 3)
                            for pipeline in ("B3", "G2")},
        "paired_median_latency_delta_G2_minus_B3_ms": statistics.median(
            value["G2"] - value["B3"] for value in paired.values()),
        "graph_evidence_rate_G2": round(sum(any(e["type"] == "graph" for e in item["result"]["evidence"])
                                            for item in results if item["pipeline_id"] == "G2") / len(ids), 6),
        "interpretation": "Flow/latency smoke only" if generator == "mock" else "Candidate outputs require judge/human review"}
    (ROOT / f"data/manifests/{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
