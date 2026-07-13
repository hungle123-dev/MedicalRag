"""Blinded direct scoring and paired control analysis for a cached E5 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.judge import GatewayJudge, correctness_input, faithfulness_input
from app.controls import matched_graph_control_audit
from statistics import paired_bootstrap


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(rows: list[dict], key: str) -> float:
    return round(sum(float(row[key]) for row in rows) / len(rows), 4)


def win_tie_loss(left: list[float], right: list[float]) -> dict:
    deltas = [b - a for a, b in zip(left, right)]
    return {"right_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "right_losses": sum(delta < 0 for delta in deltas)}


def comparison(rows: list[dict], left: str, right: str) -> dict:
    paired = {row["question_id"]: {} for row in rows}
    for row in rows:
        if row["pipeline_id"] in paired[row["question_id"]]:
            raise RuntimeError(f"Duplicate judge record for {row['question_id']} {row['pipeline_id']}")
        paired[row["question_id"]][row["pipeline_id"]] = row
    incomplete = [question_id for question_id, pair in paired.items()
                  if (left in pair) != (right in pair)]
    if incomplete:
        raise RuntimeError(f"Unpaired comparison {left}/{right}: {incomplete[:5]}")
    ordered = [pair for pair in paired.values() if left in pair and right in pair]
    left_scores = [pair[left]["correctness"]["correctness"] for pair in ordered]
    right_scores = [pair[right]["correctness"]["correctness"] for pair in ordered]
    deltas = [right - left for left, right in zip(left_scores, right_scores)]
    mean = sum(deltas) / max(1, len(deltas))
    paired_sd = (math.sqrt(sum((delta - mean) ** 2 for delta in deltas) / (len(deltas) - 1))
                 if len(deltas) > 1 else 0.0)
    standard_error = paired_sd / math.sqrt(max(1, len(deltas)))
    bootstrap = paired_bootstrap(left_scores, right_scores)
    return {"questions": len(ordered), "paired_bootstrap": bootstrap,
            "win_tie_loss": win_tie_loss(left_scores, right_scores),
            "precision": {
                "observed_paired_sd": round(paired_sd, 6),
                "standard_error": round(standard_error, 6),
                "bootstrap_ci_width": round(bootstrap["ci95"][1] - bootstrap["ci95"][0], 6),
                "normal_approx_half_width_95": round(1.96 * standard_error, 6),
                "projected_eval340_half_width_95": round(1.96 * paired_sd / math.sqrt(340), 6),
                "projected_eval340_mde_80pct_power_two_sided": round(
                    (1.96 + 0.84) * paired_sd / math.sqrt(340), 6),
                "interpretation": "planning approximation from observed paired SD; not an equivalence margin",
            }}


def comparisons(rows: list[dict]) -> dict:
    return {key: comparison(rows, left, right) for key, left, right in (
        ("G2_minus_B3", "B3", "G2"),
        ("G2_minus_X1_extra_text", "X1", "G2"),
        ("G2_minus_X2_random_path", "X2", "G2"),
    )}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    evaluation_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                       text=True, capture_output=True).stdout.strip()
    evaluation_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                            check=True, text=True, capture_output=True).stdout)
    locked = manifest.get("split") == "eval" or manifest.get("locked_execution", False)
    if locked and args.force:
        raise SystemExit("Locked evaluation forbids --force")
    if locked:
        if evaluation_dirty:
            raise SystemExit("Locked evaluation requires a clean worktree")
    arms = manifest["arms"]
    judge = GatewayJudge(ROOT)
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text(encoding="utf-8"))
    if locked and judge.model != protocol["judge"]["model"]:
        raise SystemExit("Locked judge does not match configs/protocol.yaml")
    if manifest.get("generator_model") == judge.model:
        raise RuntimeError("Generator and judge models must differ")
    current_hashes = {
        "experiments": sha(ROOT / "configs/experiments.yaml"),
        "pipelines": sha(ROOT / "configs/pipelines.yaml"),
        "protocol": sha(ROOT / "configs/protocol.yaml"),
        "prompt": sha(ROOT / "configs/prompts/answer_v1.txt"),
        "data": sha(ROOT / "data/manifests/files.json"),
        "population": sha(ROOT / f"data/manifests/bioasq_{manifest['split']}_question_ids.json"),
    }
    if current_hashes != manifest["config_hashes"]:
        raise RuntimeError("Current frozen configs/population do not match the generation manifest")
    judge_prompt_hashes = {
        "correctness": sha(ROOT / "configs/prompts/judge_correctness_completeness_v1.txt"),
        "faithfulness": sha(ROOT / "configs/prompts/judge_faithfulness_citation_v1.txt")}
    snapshots = {question_id: json.loads(
        (run / f"{question_id}_evidence.json").read_text(encoding="utf-8"))
        for question_id in manifest["question_ids"]}
    x2_audits = {}
    for question_id, snapshot in snapshots.items():
        targets = [item for item in snapshot["arms"]["G2"]["evidence"]
                   if item.get("type") == "graph"]
        controls = [item for item in snapshot["arms"]["X2"]["evidence"]
                    if item.get("matched_target_id") is not None]
        forbidden = {str(node["id"]) for item in targets for node in item.get("nodes", [])}
        forbidden.update(str(item["id"]) for item in snapshot["arms"]["G2"].get("linked", []))
        x2_audits[question_id] = matched_graph_control_audit(targets, controls, forbidden)

    def score_arm(question_id: str, arm: str) -> dict:
        target = run / f"{question_id}_{arm}.json"
        item = json.loads(target.read_text(encoding="utf-8"))
        judge_target = run / f"{question_id}_{arm}.judge.json"
        snapshot = snapshots[question_id]
        retrieval_positive = bool(snapshot["arms"]["G2"].get("graph_retrieval_positive"))
        control_complete = (x2_audits[question_id]["complete"] if arm == "X2" else
                            bool(snapshot["arms"][arm].get("control_complete", True)))
        correctness_payload = faithfulness_payload = None
        if item.get("status", "completed") == "completed":
            result = item["result"]
            correctness_payload = correctness_input(
                item["question"], item["reference_answer"], result["answer"], result["evidence"])
            faithfulness_payload = faithfulness_input(result["answer"], result["citations"])
        evaluation_input = {
            "question_id": question_id, "arm": arm, "item_sha256": sha(target),
            "correctness_payload": correctness_payload, "faithfulness_payload": faithfulness_payload,
            "judge_model": judge.model, "judge_endpoint": judge.base_url,
            "judge_prompt_hashes": judge_prompt_hashes,
        }
        evaluation_input_hash = hashlib.sha256(json.dumps(
            evaluation_input, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if judge_target.exists() and not args.force:
            cached_record = json.loads(judge_target.read_text(encoding="utf-8"))
            if cached_record.get("evaluation_input_hash") != evaluation_input_hash:
                raise RuntimeError(f"Stale judge record requires explicit regeneration: {judge_target.name}")
            return cached_record
        if item.get("status", "completed") != "completed":
            record = {"question_id": question_id, "pipeline_id": arm, "status": "failed",
                      "question_type": snapshot.get("question_type"),
                      "graph_retrieval_positive": retrieval_positive,
                      "control_complete": control_complete,
                      "citation_integrity": {"valid": False},
                      "correctness": {"correctness": 0, "completeness": 0, "confidence": 0,
                                      "claims": [], "justification": "Generation failure"},
                      "faithfulness": {"citation_precision": 0, "citation_recall": 0,
                                       "unsupported_claim_rate": 1, "invented_citation_rate": 0,
                                       "confidence": 0, "claims": [], "justification": "Generation failure"}}
        else:
            correctness = judge.evaluate(correctness_payload)
            faithfulness = judge.evaluate(faithfulness_payload)
            judge_metadata = {"correctness": correctness.pop("_judge"),
                              "faithfulness": faithfulness.pop("_judge")}
            record = {"question_id": question_id, "pipeline_id": arm, "status": "completed",
                      "question_type": snapshot.get("question_type") or item.get("question_type"),
                      "graph_retrieval_positive": retrieval_positive,
                      "control_complete": control_complete,
                      "citation_integrity": result["details"]["citation_integrity"],
                      "correctness": correctness, "faithfulness": faithfulness,
                      "judge_metadata": judge_metadata}
        record["evaluation_input_hash"] = evaluation_input_hash
        temporary = judge_target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(judge_target)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for question_index, question_id in enumerate(manifest["question_ids"]):
            offset = question_index % len(arms)
            order = arms[offset:] + arms[:offset]
            records.extend(pool.map(lambda arm: score_arm(question_id, arm), order))
            print(f"judged_questions={question_index + 1}/{len(manifest['question_ids'])}", flush=True)
    aggregate = {}
    for arm in arms:
        rows = [row for row in records if row["pipeline_id"] == arm]
        aggregate[arm] = {
            "n": len(rows), "failure_rate": round(sum(row["status"] == "failed" for row in rows) / len(rows), 4),
            "correctness_mean_0_2": average([row["correctness"] for row in rows], "correctness"),
            "completeness_mean_0_2": average([row["correctness"] for row in rows], "completeness"),
            "citation_precision_mean": average([row["faithfulness"] for row in rows], "citation_precision"),
            "citation_recall_mean": average([row["faithfulness"] for row in rows], "citation_recall"),
            "unsupported_claim_rate_mean": average([row["faithfulness"] for row in rows], "unsupported_claim_rate"),
            "no_invented_citation_id_rate": round(
                sum(row["citation_integrity"]["valid"] for row in rows) / len(rows), 4),
        }
    overall = comparisons(records)
    graph_ids = {row["question_id"] for row in records
                 if row["pipeline_id"] == "G2" and row["graph_retrieval_positive"]}
    graph_rows = [row for row in records if row["question_id"] in graph_ids]
    retrieval_positive = comparisons(graph_rows) if graph_rows else {}
    complete_x2_ids = {question_id for question_id, snapshot in snapshots.items()
                       if snapshot["arms"]["G2"].get("graph_retrieval_positive")
                       and x2_audits[question_id]["complete"]}
    complete_x2_rows = [row for row in records if row["question_id"] in complete_x2_ids]
    complete_x2 = (comparison(complete_x2_rows, "X2", "G2") if complete_x2_rows else None)
    requested_slots = sum(audit["requested_slots"] for audit in x2_audits.values())
    matched_slots = sum(audit["matched_slots"] for audit in x2_audits.values())
    complete_coverage = len(complete_x2_ids) / max(1, len(graph_ids))
    analysis_rules = protocol.get("analysis", {})
    complete_minimum = int(analysis_rules.get("complete_x2_min_questions", 30))
    coverage_minimum = float(analysis_rules.get("complete_x2_min_coverage", 0.8))
    lower_bounds = [overall[key]["paired_bootstrap"]["ci95"][0] for key in overall]
    sensitivity_lower = complete_x2["paired_bootstrap"]["ci95"][0] if complete_x2 else float("-inf")
    strata = {}
    for question_type in sorted({snapshot.get("question_type") for snapshot in snapshots.values()} - {None}):
        selected = [row for row in records if row.get("question_type") == question_type]
        strata[f"question_type:{question_type}"] = comparison(selected, "B3", "G2")
    for positive in (False, True):
        selected = [row for row in records if row["graph_retrieval_positive"] is positive]
        if selected:
            strata[f"graph_retrieval_positive:{str(positive).lower()}"] = comparison(selected, "B3", "G2")
    labels_by_question: dict[str, set[str]] = {}
    for question_id, snapshot in snapshots.items():
        graph_items = [item for item in snapshot["arms"]["G2"]["evidence"] if item["type"] == "graph"]
        labels = labels_by_question.setdefault(question_id, set())
        if graph_items:
            labels.add(f"max_hops:{max(item.get('hop_count', 0) for item in graph_items)}")
        for entity in snapshot["arms"]["G2"].get("linked", []):
            labels.add(f"linked_entity_type:{entity.get('type', 'unknown')}")
        for item in graph_items:
            for edge in item.get("edges", []):
                labels.add(f"relation:{edge.get('relation', 'unknown')}")
        if snapshot["arms"]["G2"].get("graph_retrieval_positive"):
            labels.add(f"x2_control_complete:{str(x2_audits[question_id]['complete']).lower()}")
    for label in sorted({label for labels in labels_by_question.values() for label in labels}):
        ids_for_label = {question_id for question_id, labels in labels_by_question.items() if label in labels}
        if len(ids_for_label) >= 5:
            selected = [row for row in records if row["question_id"] in ids_for_label]
            strata[label] = comparison(selected, "B3", "G2")
    errors = {
        "OUTPUT_FAILURE": sum(row["status"] == "failed" for row in records),
        "CITATION_INTEGRITY": sum(not row["citation_integrity"]["valid"] for row in records),
        "ANSWER_INCORRECT": sum(row["correctness"]["correctness"] == 0 for row in records),
        "ANSWER_PARTIAL": sum(row["correctness"]["correctness"] == 1 or
                              row["correctness"]["completeness"] < 2 for row in records),
        "UNSUPPORTED_CLAIM": sum(row["faithfulness"]["unsupported_claim_rate"] > 0 for row in records),
    }
    observed_judge_models = sorted({metadata.get("response_model")
                                    for row in records if row.get("status") == "completed"
                                    for metadata in row.get("judge_metadata", {}).values()
                                    if metadata.get("response_model")})
    observed_judge_fingerprints = sorted({metadata.get("system_fingerprint")
                                          for row in records if row.get("status") == "completed"
                                          for metadata in row.get("judge_metadata", {}).values()
                                          if metadata.get("system_fingerprint")})
    if len(observed_judge_models) != 1:
        raise RuntimeError(f"Judge response model drift detected: {observed_judge_models}")
    complete_sensitivity_eligible = (len(complete_x2_ids) >= complete_minimum and
                                     complete_coverage >= coverage_minimum)
    summary = {
        "run_id": manifest["run_id"], "track": "bioasq_e5_blinded_machine_judge",
        "split": manifest["split"], "generator_model": manifest["generator_model"],
        "judge_model": judge.model, "observed_judge_response_models": observed_judge_models,
        "observed_judge_system_fingerprints": observed_judge_fingerprints,
        "judge_method": "reference-based direct scoring; IDs and modality cues masked",
        "questions": len(manifest["question_ids"]),
        "evaluation_code_commit": evaluation_commit,
        "evaluation_script_sha256": sha(Path(__file__)),
        "evaluation_worktree_dirty_at_start": evaluation_dirty,
        "graph_retrieval_positive_questions": len(graph_ids),
        "aggregate": aggregate, "overall_comparisons": overall,
        "graph_retrieval_positive_secondary": retrieval_positive,
        "complete_x2_sensitivity": complete_x2,
        "control_audit": {"x2_requested_slots": requested_slots, "x2_matched_slots": matched_slots,
                           "x2_complete_questions": len(complete_x2_ids),
                           "x2_incomplete_questions": len(graph_ids - complete_x2_ids),
                           "x2_incomplete_question_ids": sorted(graph_ids - complete_x2_ids),
                           "x2_complete_coverage": round(complete_coverage, 6),
                           "minimum_complete_questions": complete_minimum,
                           "minimum_complete_coverage": coverage_minimum,
                           "complete_sensitivity_eligible": complete_sensitivity_eligible,
                           "method": "post_budget_ids_hops_canonical_path_node_disjoint_v2"},
        "stratified_g2_minus_b3": strata, "machine_error_triage": errors,
        "machine_graph_benefit_gate_passed": all(bound > 0 for bound in lower_bounds)
                                               and sensitivity_lower > 0
                                               and complete_sensitivity_eligible,
        "config_hashes": manifest["config_hashes"],
        "judge_prompt_hashes": judge_prompt_hashes,
        "strata_status": "exploratory; overlapping groups, small cells, no multiplicity-adjusted claims",
        "warning": "Machine judge is exploratory until judge-human weighted agreement reaches 0.60."
    }
    output = ROOT / f"data/manifests/{manifest['run_id']}_machine_judge.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
