"""Validate blinded reviews, quantify agreement/effects, and emit adjudication work."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from statistics import confusion_matrix, paired_bootstrap, weighted_kappa

HARM = {"none": 0, "minor": 1, "major": 2}
USEFULNESS = {"supports", "partial", "irrelevant", "misleading", "not_applicable"}


def load(packet: Path, mapping: Path) -> dict:
    assignments = {row["question_id"]: row
                   for row in json.loads(mapping.read_text(encoding="utf-8"))}
    output = {}
    with packet.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assigned = assignments[row["question_id"]]
            arms = {}
            for slot in ("a", "b"):
                try:
                    correctness = int(row[f"correctness_{slot}_0_2"])
                    completeness = int(row[f"completeness_{slot}_0_2"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Missing numeric score for {row['question_id']} slot {slot}") from exc
                harm = row[f"medical_harm_{slot}"].strip().casefold()
                usefulness = row[f"graph_usefulness_{slot}"].strip().casefold()
                if correctness not in range(3) or completeness not in range(3):
                    raise ValueError("Correctness/completeness must be 0, 1, or 2")
                if harm not in HARM or usefulness not in USEFULNESS:
                    raise ValueError(f"Invalid harm/usefulness label for {row['question_id']}")
                arms[assigned[slot]] = {
                    "correctness": correctness, "completeness": completeness,
                    "harm": harm, "graph_usefulness": usefulness,
                    "error_code": row[f"error_code_{slot}"].strip() or "NONE",
                }
            preference = row["pair_preference"].strip().casefold()
            if preference not in {"a", "tie", "b"}:
                raise ValueError(f"Invalid pair preference for {row['question_id']}")
            output[row["question_id"]] = {"arms": arms,
                "preferred_arm": "tie" if preference == "tie" else assigned[preference]}
    return output


def adjudication_rows(a: dict, b: dict) -> list[dict]:
    rows = []
    for question_id in sorted(a):
        disagreement = any(
            a[question_id]["arms"][arm][field] != b[question_id]["arms"][arm][field]
            for arm in ("B3", "G2") for field in ("correctness", "completeness", "harm")
        ) or a[question_id]["preferred_arm"] != b[question_id]["preferred_arm"]
        if disagreement:
            rows.append({"question_id": question_id,
                         "b3_correctness_0_2": "", "g2_correctness_0_2": "",
                         "b3_completeness_0_2": "", "g2_completeness_0_2": "",
                         "b3_harm": "", "g2_harm": "", "preferred_arm": "",
                         "adjudication_rationale": ""})
    return rows


def load_adjudication(path: Path | None) -> dict | None:
    if path is None:
        return None
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output[row["question_id"]] = {arm: {
                "correctness": int(row[f"{arm.casefold()}_correctness_0_2"]),
                "completeness": int(row[f"{arm.casefold()}_completeness_0_2"]),
                "harm": row[f"{arm.casefold()}_harm"].strip().casefold(),
            } for arm in ("B3", "G2")}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer_a", type=Path); parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("mapping_a", type=Path); parser.add_argument("mapping_b", type=Path)
    parser.add_argument("--run", type=Path, help="Run directory containing optional machine-judge records")
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/human_review/analysis.json")
    args = parser.parse_args()
    a, b = load(args.reviewer_a, args.mapping_a), load(args.reviewer_b, args.mapping_b)
    if set(a) != set(b):
        raise SystemExit("Reviewer populations differ")

    agreement = {}
    for arm in ("B3", "G2"):
        agreement[arm] = {}
        for field in ("correctness", "completeness"):
            left = [a[qid]["arms"][arm][field] for qid in sorted(a)]
            right = [b[qid]["arms"][arm][field] for qid in sorted(b)]
            agreement[arm][field] = {
                "weighted_kappa": weighted_kappa(left, right),
                "confusion_matrix_rows_A_columns_B": confusion_matrix(left, right),
            }
        left_harm = [HARM[a[qid]["arms"][arm]["harm"]] for qid in sorted(a)]
        right_harm = [HARM[b[qid]["arms"][arm]["harm"]] for qid in sorted(b)]
        agreement[arm]["harm"] = {"weighted_kappa": weighted_kappa(left_harm, right_harm),
                                    "confusion_matrix_rows_A_columns_B": confusion_matrix(left_harm, right_harm)}

    rows = adjudication_rows(a, b)
    adjudication_target = args.output.parent / "adjudication_required.csv"
    adjudication_target.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with adjudication_target.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)

    adjudicated_rows = load_adjudication(args.adjudication)
    if adjudicated_rows is not None:
        disputed = {row["question_id"] for row in rows}
        missing = disputed - set(adjudicated_rows)
        if missing:
            raise ValueError(f"Adjudication is missing {len(missing)} disputed questions")
        consensus = {question_id: adjudicated_rows.get(question_id) or {
            arm: {field: a[question_id]["arms"][arm][field]
                  for field in ("correctness", "completeness", "harm")}
            for arm in ("B3", "G2")
        } for question_id in a}
    else:
        consensus = {question_id: {arm: {
        "correctness": (a[question_id]["arms"][arm]["correctness"] +
                        b[question_id]["arms"][arm]["correctness"]) / 2,
        "completeness": (a[question_id]["arms"][arm]["completeness"] +
                         b[question_id]["arms"][arm]["completeness"]) / 2,
        "harm": "major" if "major" in {a[question_id]["arms"][arm]["harm"], b[question_id]["arms"][arm]["harm"]}
                else "minor" if "minor" in {a[question_id]["arms"][arm]["harm"], b[question_id]["arms"][arm]["harm"]}
                else "none",
        } for arm in ("B3", "G2")} for question_id in a}
    qids = sorted(consensus)
    correctness_effect = paired_bootstrap(
        [consensus[qid]["B3"]["correctness"] for qid in qids],
        [consensus[qid]["G2"]["correctness"] for qid in qids])
    harm_effect = paired_bootstrap(
        [float(consensus[qid]["B3"]["harm"] != "none") for qid in qids],
        [float(consensus[qid]["G2"]["harm"] != "none") for qid in qids])

    adjudication_complete = adjudicated_rows is not None or not rows
    judge_validity = {"status": "requires_adjudication_and_machine_judge_records"}
    if adjudication_complete and args.run:
        judge_validity = {}
        for arm in ("B3", "G2"):
            machine = [json.loads((args.run / f"{qid}_{arm}.judge.json").read_text(encoding="utf-8"))
                       ["correctness"]["correctness"] for qid in qids]
            human = [int(consensus[qid][arm]["correctness"]) for qid in qids]
            judge_validity[arm] = {"weighted_kappa": weighted_kappa(human, machine),
                                   "confusion_matrix_rows_human_columns_judge": confusion_matrix(human, machine)}

    all_kappas = [agreement[arm][field]["weighted_kappa"]
                  for arm in ("B3", "G2") for field in ("correctness", "completeness")]
    result = {
        "questions": len(qids), "agreement": agreement,
        "disagreement_questions": len(rows),
        "adjudication_status": "complete" if adjudicated_rows is not None else "required" if rows else "not_needed",
        "paired_human_correctness_g2_minus_b3": correctness_effect,
        "paired_harm_rate_g2_minus_b3": harm_effect,
        "graph_usefulness_g2": dict(Counter(
            review[qid]["arms"]["G2"]["graph_usefulness"] for review in (a, b) for qid in qids)),
        "error_taxonomy": dict(Counter(
            review[qid]["arms"][arm]["error_code"] for review in (a, b)
            for qid in qids for arm in ("B3", "G2"))),
        "judge_human_validity": judge_validity,
        "human_gate_passed": adjudication_complete and min(all_kappas) >= .60
                             and harm_effect["ci95"][1] <= 0,
        "note": "Without adjudication, paired effects use reviewer means and are exploratory only.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
