"""Blinded machine-judge evaluation for a cached paired B3/G2 run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.judge import GatewayJudge, correctness_input, faithfulness_input
from statistics import paired_bootstrap


def average(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(sum(values) / len(values), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    judge = GatewayJudge(ROOT)
    if manifest.get("generator_model") == judge.model:
        raise RuntimeError("Generator and judge models must differ")
    records = []
    for target in sorted(run.glob("*_B3.json")) + sorted(run.glob("*_G2.json")):
        item = json.loads(target.read_text(encoding="utf-8"))
        result = item["result"]
        correctness = judge.evaluate(correctness_input(item["question"], item["reference_answer"], result["answer"]))
        faithfulness = judge.evaluate(faithfulness_input(result["answer"], result["citations"]))
        record = {"question_id": item["question_id"], "pipeline_id": item["pipeline_id"],
                  "graph_evidence": any(evidence["type"] == "graph" for evidence in result["evidence"]),
                  "citation_integrity": result["details"]["citation_integrity"],
                  "correctness": correctness, "faithfulness": faithfulness}
        records.append(record)
        (run / f"{target.stem}.judge.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    by_pipeline = {pipeline: [row for row in records if row["pipeline_id"] == pipeline]
                   for pipeline in ("B3", "G2")}
    aggregate = {}
    for pipeline, rows in by_pipeline.items():
        aggregate[pipeline] = {
            "correctness_mean_0_2": average([row["correctness"] for row in rows], "correctness"),
            "completeness_mean_0_2": average([row["correctness"] for row in rows], "completeness"),
            "citation_precision_mean": average([row["faithfulness"] for row in rows], "citation_precision"),
            "unsupported_claim_rate_mean": average([row["faithfulness"] for row in rows], "unsupported_claim_rate"),
            "citation_integrity_rate": round(sum(row["citation_integrity"]["valid"] for row in rows) / len(rows), 4),
        }
    paired = {row["question_id"]: {} for row in records}
    for row in records:
        paired[row["question_id"]][row["pipeline_id"]] = row
    ordered = [paired[question_id] for question_id in manifest["question_ids"]]
    summary = {"run_id": manifest["run_id"], "track": "exploratory_dev_machine_judge",
               "generator_model": manifest["generator_model"], "judge_model": judge.model,
               "questions": len(ordered), "graph_positive_questions": sum(row["G2"]["graph_evidence"] for row in ordered),
               "aggregate": aggregate,
               "paired_correctness_G2_minus_B3": paired_bootstrap(
                   [row["B3"]["correctness"]["correctness"] for row in ordered],
                   [row["G2"]["correctness"]["correctness"] for row in ordered]),
               "warning": "Exploratory LLM-judge results; locked test and qualified human review are still required."}
    output = ROOT / f"data/manifests/{manifest['run_id']}_machine_judge.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
