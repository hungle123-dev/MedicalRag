"""Export randomized, pipeline-blinded B3/G2 packets for qualified reviewers."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def human_bundle(result: dict) -> tuple[str, str]:
    evidence = result["evidence"]
    mapping = {item["id"]: f"E{index}" for index, item in enumerate(evidence, start=1)}
    answer = result["answer"]
    for source, target in sorted(mapping.items(), key=lambda item: -len(item[0])):
        answer = answer.replace(source, target)
    bundle = [{"id": mapping[item["id"]],
               "kind": "structured_graph" if item["type"] == "graph" else "literature",
               "title": item.get("title", ""), "snippet": item.get("snippet", "")}
              for item in evidence]
    return answer, json.dumps(bundle, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reviewer", choices=("a", "b"), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    frozen = json.loads((ROOT / "data/manifests/bioasq_human_100.json").read_text(encoding="utf-8"))
    rng = random.Random(frozen["seed"] + (1 if args.reviewer == "a" else 2))
    output = ROOT / "artifacts/human_review"; output.mkdir(parents=True, exist_ok=True)
    rows, mapping_rows = [], []
    for question_id in frozen["ids"]:
        items = {arm: json.loads((run / f"{question_id}_{arm}.json").read_text(encoding="utf-8"))
                 for arm in ("B3", "G2")}
        if any(item.get("status") != "completed" for item in items.values()):
            raise RuntimeError(f"Missing completed B3/G2 output for {question_id}")
        first, second = (("B3", "G2") if rng.random() < 0.5 else ("G2", "B3"))
        answer_a, evidence_a = human_bundle(items[first]["result"])
        answer_b, evidence_b = human_bundle(items[second]["result"])
        rows.append({"question_id": question_id, "question": items[first]["question"],
                     "reference_answer": items[first]["reference_answer"],
                     "answer_a": answer_a, "evidence_a": evidence_a,
                     "answer_b": answer_b, "evidence_b": evidence_b,
                     "correctness_a_0_2": "", "completeness_a_0_2": "",
                     "graph_usefulness_a": "", "error_code_a": "", "rationale_a": "",
                     "correctness_b_0_2": "", "completeness_b_0_2": "",
                     "graph_usefulness_b": "", "error_code_b": "", "rationale_b": ""})
        mapping_rows.append({"question_id": question_id, "a": first, "b": second})
    packet = output / f"reviewer_{args.reviewer}.csv"
    with packet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (output / f"reviewer_{args.reviewer}_mapping.json").write_text(
        json.dumps(mapping_rows, indent=2), encoding="utf-8")
    print(packet)


if __name__ == "__main__":
    main()
