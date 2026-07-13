import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from analyze_human_review import adjudication_rows, load


def test_human_review_unblinds_and_detects_disagreement(tmp_path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([{"question_id": "q1", "a": "G2", "b": "B3"}]))
    fields = ["question_id", "correctness_a_0_2", "completeness_a_0_2",
              "graph_usefulness_a", "medical_harm_a", "error_code_a",
              "correctness_b_0_2", "completeness_b_0_2",
              "graph_usefulness_b", "medical_harm_b", "error_code_b", "pair_preference"]

    def packet(name, correctness):
        target = tmp_path / name
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerow({"question_id": "q1", "correctness_a_0_2": correctness,
                             "completeness_a_0_2": 2, "graph_usefulness_a": "supports",
                             "medical_harm_a": "none", "error_code_a": "NONE",
                             "correctness_b_0_2": 1, "completeness_b_0_2": 1,
                             "graph_usefulness_b": "not_applicable", "medical_harm_b": "none",
                             "error_code_b": "NONE", "pair_preference": "a"})
        return target

    reviewer_a = load(packet("a.csv", 2), mapping)
    reviewer_b = load(packet("b.csv", 1), mapping)
    assert reviewer_a["q1"]["arms"]["G2"]["correctness"] == 2
    assert adjudication_rows(reviewer_a, reviewer_b)[0]["question_id"] == "q1"
