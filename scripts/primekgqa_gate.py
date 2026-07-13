"""Validate PrimeKGQA SPARQL denotations against the pinned PrimeKG CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import ijson


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifests" / "primekgqa_compatibility.json"
TRIPLE = re.compile(
    r"(?P<s><https://zitniklab\.hms\.harvard\.edu/projects/PrimeKG/node/(?P<sid>\d+)>|\?uri)\s+"
    r"(?P<p><https://zitniklab\.hms\.harvard\.edu/projects/PrimeKG/vocab/(?P<rel>[^>]+)>|\?uri)\s+"
    r"(?P<o><https://zitniklab\.hms\.harvard\.edu/projects/PrimeKG/node/(?P<oid>\d+)>|\?uri)\s*\."
)


def patterns(sparql: str) -> list[tuple[str | None, str | None, str | None]]:
    return [
        (match.group("sid"), unquote(match.group("rel")) if match.group("rel") else None, match.group("oid"))
        for match in TRIPLE.finditer(sparql)
    ]


def execute(
    query: list[tuple[str | None, str | None, str | None]],
    outgoing: dict[tuple[str, str], set[str]],
    incoming: dict[tuple[str, str], set[str]],
    between: dict[tuple[str, str], set[str]],
) -> tuple[str, set[str]]:
    candidates: list[set[str]] = []
    answer_kind = "unknown"
    for subject, relation, obj in query:
        missing = sum(value is None for value in (subject, relation, obj))
        if missing == 0:
            if obj not in outgoing.get((subject, relation), set()):
                return answer_kind, set()
        elif missing != 1:
            raise ValueError("Only one ?uri variable per triple is supported")
        elif subject is None:
            answer_kind = "node"
            candidates.append(incoming.get((relation, obj), set()))
        elif relation is None:
            answer_kind = "relation"
            candidates.append(between.get((subject, obj), set()))
        else:
            answer_kind = "node"
            candidates.append(outgoing.get((subject, relation), set()))
    return answer_kind, set.intersection(*candidates) if candidates else set()


def load_sample(count: int, seed: int) -> list[dict]:
    path = RAW / "primekgqa" / "val_call_bioLLM.json"
    eligible = []
    with path.open("rb") as stream:
        for row in ijson.items(stream, "item"):
            parsed = patterns(str(row.get("sparql", "")))
            if row.get("generated_question") and parsed and all(sum(v is None for v in triple) <= 1 for triple in parsed):
                eligible.append(row)
    return random.Random(seed).sample(eligible, min(count, len(eligible)))


def run(count: int, seed: int) -> dict:
    sample = load_sample(count, seed)
    parsed = [patterns(str(row["sparql"])) for row in sample]
    constants = {value for query in parsed for triple in query for value in (triple[0], triple[2]) if value}
    outgoing: dict[tuple[str, str], set[str]] = defaultdict(set)
    incoming: dict[tuple[str, str], set[str]] = defaultdict(set)
    between: dict[tuple[str, str], set[str]] = defaultdict(set)
    with (RAW / "primekg" / "edges.csv").open(encoding="utf-8", newline="") as stream:
        for edge in csv.DictReader(stream):
            x, y, relation = edge["x_index"], edge["y_index"], edge["display_relation"]
            if x in constants or y in constants:
                outgoing[(x, relation)].add(y)
                incoming[(relation, y)].add(x)
                between[(x, y)].add(relation)
    names = {}
    with (RAW / "primekg" / "nodes.tab").open(encoding="utf-8", newline="") as stream:
        for node in csv.DictReader(stream, delimiter="\t"):
            names[node["node_index"]] = node["node_name"]
    records = []
    for row, query in zip(sample, parsed):
        try:
            kind, result = execute(query, outgoing, incoming, between)
            predicted = {names.get(value, value) for value in result} if kind == "node" else result
            gold = {unquote(str(value)).strip("<>") for value in row.get("answer_sparql", [])}
            executable = bool(predicted)
            records.append({
                "id": row.get("id"), "type": row.get("type"), "answer_kind": kind,
                "executable": executable, "exact": executable and predicted == gold,
                "predicted_count": len(predicted), "gold_count": len(gold),
            })
        except ValueError as exc:
            records.append({"id": row.get("id"), "executable": False, "error": str(exc), "exact": False})
    executable = sum(record["executable"] for record in records)
    exact = sum(record["exact"] for record in records)
    report = {
        "split": "validation", "seed": seed, "sample_size": len(records),
        "executable_rate": executable / max(len(records), 1),
        "execution_exact_match": exact / max(len(records), 1),
        "gate_threshold": 0.99, "gate_passed": executable / max(len(records), 1) >= 0.99,
        "diagnosis": "PrimeKGQA RDF node IRIs do not map directly to the pinned Dataverse CSV node_index when denotations are empty.",
        "fallback": "Evaluate normalized graph patterns from the dataset value field until the matching RDF mapping is available.",
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    report = run(args.count, args.seed)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
