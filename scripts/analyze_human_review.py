"""Validate two completed reviewer packets and compute independent agreement."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from statistics import confusion_matrix, weighted_kappa


def load(packet: Path, mapping: Path) -> dict:
    assignments = {row["question_id"]: row for row in json.loads(mapping.read_text(encoding="utf-8"))}
    output = {}
    with packet.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            assigned = assignments[row["question_id"]]
            output[row["question_id"]] = {
                assigned[slot]: {"correctness": int(row[f"correctness_{slot}_0_2"]),
                                  "completeness": int(row[f"completeness_{slot}_0_2"])}
                for slot in ("a", "b")}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer_a", type=Path); parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("mapping_a", type=Path); parser.add_argument("mapping_b", type=Path)
    args = parser.parse_args()
    a, b = load(args.reviewer_a, args.mapping_a), load(args.reviewer_b, args.mapping_b)
    if set(a) != set(b): raise SystemExit("Reviewer populations differ")
    result = {}
    for arm in ("B3", "G2"):
        left = [a[qid][arm]["correctness"] for qid in sorted(a)]
        right = [b[qid][arm]["correctness"] for qid in sorted(b)]
        result[arm] = {"weighted_kappa_correctness": weighted_kappa(left, right),
                       "confusion_matrix_rows_A_columns_B": confusion_matrix(left, right)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
