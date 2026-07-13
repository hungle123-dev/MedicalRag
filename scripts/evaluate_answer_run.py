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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    arms = manifest["arms"]
    judge = GatewayJudge(ROOT)
    if manifest.get("generator_model") == judge.model:
        raise RuntimeError("Generator and judge models must differ")
    records = []
    for question_index, question_id in enumerate(manifest["question_ids"]):
        offset = question_index % len(arms)
        order = arms[offset:] + arms[:offset]
        for arm in order:
            target = run / f"{question_id}_{arm}.json"
            item = json.loads(target.read_text(encoding="utf-8"))
            if item.get("status", "completed") != "completed":
                record = {"question_id": question_id, "pipeline_id": arm, "status": "failed",
                          "graph_positive": False, "citation_integrity": {"valid": False},
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
                          "graph_positive": bool(result["details"].get("graph_positive")),
                          "control_complete": result["details"].get("control_complete", True),
                          "citation_integrity": result["details"]["citation_integrity"],
                          "correctness": correctness, "faithfulness": faithfulness}
            records.append(record)
            (run / f"{question_id}_{arm}.judge.json").write_text(
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
            "citation_integrity_rate": round(sum(row["citation_integrity"]["valid"] for row in rows) / len(rows), 4),
        }
    overall = {"G2_minus_B3": comparison(records, "B3", "G2"),
               "G2_minus_X1_extra_text": comparison(records, "X1", "G2"),
               "G2_minus_X2_random_path": comparison(records, "X2", "G2")}
    graph_ids = {row["question_id"] for row in records
                 if row["pipeline_id"] == "G2" and row["graph_positive"]}
    graph_rows = [row for row in records if row["question_id"] in graph_ids]
    graph_positive = ({key: comparison(graph_rows, left, right) for key, left, right in (
        ("G2_minus_B3", "B3", "G2"), ("G2_minus_X1_extra_text", "X1", "G2"),
        ("G2_minus_X2_random_path", "X2", "G2"))} if graph_rows else {})
    lower_bounds = [overall[key]["paired_bootstrap"]["ci95"][0] for key in overall]
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
        "questions": len(manifest["question_ids"]), "graph_positive_questions": len(graph_ids),
        "aggregate": aggregate, "overall_comparisons": overall,
        "graph_positive_secondary": graph_positive, "machine_error_triage": errors,
        "machine_graph_benefit_gate_passed": all(bound > 0 for bound in lower_bounds),
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
