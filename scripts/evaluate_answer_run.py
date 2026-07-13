"""Blinded direct scoring and paired control analysis for a cached E5 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.judge import GatewayJudge, correctness_input, faithfulness_input
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
        paired[row["question_id"]][row["pipeline_id"]] = row
    ordered = [pair for pair in paired.values() if left in pair and right in pair]
    left_scores = [pair[left]["correctness"]["correctness"] for pair in ordered]
    right_scores = [pair[right]["correctness"]["correctness"] for pair in ordered]
    return {"questions": len(ordered), "paired_bootstrap": paired_bootstrap(left_scores, right_scores),
            "win_tie_loss": win_tie_loss(left_scores, right_scores)}


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
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    arms = manifest["arms"]
    judge = GatewayJudge(ROOT)
    if manifest.get("generator_model") == judge.model:
        raise RuntimeError("Generator and judge models must differ")
    snapshots = {question_id: json.loads(
        (run / f"{question_id}_evidence.json").read_text(encoding="utf-8"))
        for question_id in manifest["question_ids"]}
    records = []
    for question_index, question_id in enumerate(manifest["question_ids"]):
        offset = question_index % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            target = run / f"{question_id}_{arm}.json"
            item = json.loads(target.read_text(encoding="utf-8"))
            judge_target = run / f"{question_id}_{arm}.judge.json"
            if judge_target.exists() and not args.force:
                records.append(json.loads(judge_target.read_text(encoding="utf-8")))
                continue
            snapshot = snapshots[question_id]
            retrieval_positive = bool(snapshot["arms"]["G2"].get("graph_retrieval_positive"))
            control_complete = bool(snapshot["arms"][arm].get("control_complete", True))
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
                result = item["result"]
                correctness = judge.evaluate(correctness_input(
                    item["question"], item["reference_answer"], result["answer"], result["evidence"]))
                faithfulness = judge.evaluate(faithfulness_input(result["answer"], result["citations"]))
                record = {"question_id": question_id, "pipeline_id": arm, "status": "completed",
                          "question_type": snapshot.get("question_type") or item.get("question_type"),
                          "graph_retrieval_positive": retrieval_positive,
                          "control_complete": control_complete,
                          "citation_integrity": result["details"]["citation_integrity"],
                          "correctness": correctness, "faithfulness": faithfulness}
            records.append(record)
            judge_target.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
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
                       and snapshot["arms"]["X2"].get("control_complete", True)}
    complete_x2_rows = [row for row in records if row["question_id"] in complete_x2_ids]
    complete_x2 = (comparison(complete_x2_rows, "X2", "G2") if complete_x2_rows else None)
    requested_slots = sum(sum(item["type"] == "graph" for item in snapshot["arms"]["G2"]["evidence"])
                          for snapshot in snapshots.values())
    matched_slots = sum(sum(item.get("matched_target_id") is not None
                            for item in snapshot["arms"]["X2"]["evidence"])
                        for snapshot in snapshots.values())
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
            labels.add(f"x2_control_complete:{str(snapshot['arms']['X2'].get('control_complete', True)).lower()}")
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
    summary = {
        "run_id": manifest["run_id"], "track": "bioasq_e5_blinded_machine_judge",
        "split": manifest["split"], "generator_model": manifest["generator_model"],
        "judge_model": judge.model, "judge_method": "reference-based direct scoring; modality IDs masked",
        "questions": len(manifest["question_ids"]),
        "graph_retrieval_positive_questions": len(graph_ids),
        "aggregate": aggregate, "overall_comparisons": overall,
        "graph_retrieval_positive_secondary": retrieval_positive,
        "complete_x2_sensitivity": complete_x2,
        "control_audit": {"x2_requested_slots": requested_slots, "x2_matched_slots": matched_slots,
                          "x2_complete_questions": len(complete_x2_ids),
                          "x2_incomplete_questions": len(graph_ids - complete_x2_ids)},
        "stratified_g2_minus_b3": strata, "machine_error_triage": errors,
        "machine_graph_benefit_gate_passed": all(bound > 0 for bound in lower_bounds)
                                               and sensitivity_lower > 0,
        "config_hashes": manifest["config_hashes"],
        "judge_prompt_hashes": {
            "correctness": sha(ROOT / "configs/prompts/judge_correctness_completeness_v1.txt"),
            "faithfulness": sha(ROOT / "configs/prompts/judge_faithfulness_citation_v1.txt")},
        "warning": "Machine judge is exploratory until weighted human agreement reaches 0.60."
    }
    output = ROOT / f"data/manifests/{manifest['run_id']}_machine_judge.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
