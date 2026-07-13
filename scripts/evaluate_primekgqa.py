"""Evaluate the rule-based PrimeKG retriever using PrimeKGQA answer sets/patterns."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import ijson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.graph import PrimeKGIndex
from app.controls import matched_random_paths, no_path_reranker, one_hop

ALIASES = {
    "is associated with": "associated with", "presents the expression": "expression present",
    "does not present the expression": "expression absent", "has a side effect": "side effect",
    "is indicated for": "indication", "is contraindicated for": "contraindication",
    "is in protein-protein interaction with": "ppi", "has synergistic interactio with": "synergistic interaction",
    "presents the phenotype": "phenotype present", "absents the phenotype": "phenotype absent",
    "absents the expression": "expression absent", "has parent-child relation with": "parent-child",
    "enzymes": "enzyme", "is contraindication for": "contraindication", "has side effect of": "side effect",
    "targets": "target", "is the transporter of": "transporter", "has off-label use:": "off-label use",
    "is linked to": "linked to",
}


def norm(value) -> str:
    text = unquote(str(value)).strip().strip("[]").strip("<>").strip().casefold()
    if "/vocab/" in text: text = text.rsplit("/vocab/", 1)[1]
    if "/node/" in text: text = text.rsplit("/node/", 1)[1]
    return ALIASES.get(text, text)


def score(predicted: set[str], gold: set[str]) -> dict:
    intersection = len(predicted & gold)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(gold) if gold else 0.0
    return {"exact_match": float(predicted == gold and bool(gold)), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def answer_kind(sparql: str) -> str:
    triples = re.findall(r"([^\n.]+\?uri[^\n.]*)\.", sparql)
    return "relation" if any(re.search(r">\s+\?uri\s+<", triple) for triple in triples) else "node"


def pattern(row: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    values = row.get("value") or []
    return (tuple(sorted({norm(part) for triple in values for part in (triple[0], triple[2])})),
            tuple(sorted({norm(triple[1]) for triple in values})))


def load_rows(split: str, sample: int | None, seed: int) -> list[dict]:
    path = ROOT / f"data/raw/primekgqa/{split}_call_bioLLM.json"
    with path.open("rb") as stream:
        eligible = [row for row in ijson.items(stream, "item")
                    if str(row.get("generated_question") or "").strip() and row.get("answer_sparql")]
    return random.Random(seed).sample(eligible, min(sample, len(eligible))) if sample else eligible


def aggregate(rows: list[dict]) -> dict:
    keys = ("exact_match", "precision", "recall", "f1")
    return {key: round(sum(row[key] for row in rows) / len(rows), 6) if rows else None for key in keys} | {"n": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sample", type=int, default=300, help="0 means the complete split")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--variant", choices=("full", "one_hop", "no_path_reranker", "random_path"), default="full")
    args = parser.parse_args()
    rows = load_rows(args.split, args.sample or None, args.seed)
    graph = PrimeKGIndex(ROOT / "indexes/primekg.sqlite3")
    train_patterns = set()
    if args.split == "test":
        with (ROOT / "data/raw/primekgqa/train_call_bioLLM.json").open("rb") as stream:
            train_patterns = {pattern(row) for row in ijson.items(stream, "item")}
    records = []
    for row in rows:
        question = str(row["generated_question"])
        seeds = graph.link(question)
        paths = graph.paths(seeds, question=question)
        if args.variant == "one_hop":
            paths = one_hop(paths)
        elif args.variant in {"no_path_reranker", "random_path"}:
            pool = graph.paths(seeds, limit=200, question=question)
            paths = no_path_reranker(pool) if args.variant == "no_path_reranker" else matched_random_paths(pool, paths, args.seed + int(row.get("id", 0)))
        kind = answer_kind(str(row.get("sparql", "")))
        gold = {norm(value) for value in row.get("answer_sparql", [])}
        if kind == "relation":
            predicted = {norm(edge["relation"]) for path in paths for edge in path["edges"]}
        else:
            seed_ids = {str(seed["id"]) for seed in seeds}
            predicted = {norm(node["name"]) for path in paths for node in path["nodes"] if node["id"] not in seed_ids}
        metrics = score(predicted, gold)
        entity_pattern, relation_pattern = pattern(row)
        mentioned = {entity for entity in entity_pattern if entity and entity in norm(question) and not entity.isdigit()}
        linked = {norm(seed["name"]) for seed in seeds}
        entity_metrics = score(linked, mentioned) if mentioned else None
        predicted_relations = {norm(edge["relation"]) for path in paths for edge in path["edges"]}
        relation_gold = set(relation_pattern)
        relation_metrics = score(predicted_relations, relation_gold)
        records.append({"id": row.get("id"), "type": row.get("type"), "node_count": len(set(entity_pattern)),
            "answer_type": kind, "seen_pattern": (entity_pattern, relation_pattern) in train_patterns if train_patterns else None,
            **metrics, "entity_f1": entity_metrics["f1"] if entity_metrics else None,
            "linked_entities": sorted(linked), "gold_mentions": sorted(mentioned),
            "relation_f1": relation_metrics["f1"], "path_valid": float(bool(paths)),
            "graph_answerability": "full" if metrics["f1"] == 1 else "partial" if metrics["f1"] > 0 else "none"})
    strata = defaultdict(list)
    for record in records:
        strata[f"nodes_{record['node_count']}"] .append(record)
        strata[f"answer_{record['answer_type']}"] .append(record)
        if record["seen_pattern"] is not None: strata["seen" if record["seen_pattern"] else "unseen"].append(record)
    report = {"split": args.split, "sample_size": len(records), "seed": args.seed,
        "compatibility_gate_passed": False, "evaluation_mode": "normalized_pattern_fallback", "variant": args.variant,
        "answer_set": aggregate(records),
        "entity_link_f1": round(sum(r["entity_f1"] for r in records if r["entity_f1"] is not None) /
                                max(sum(r["entity_f1"] is not None for r in records), 1), 6),
        "relation_f1": round(sum(r["relation_f1"] for r in records) / len(records), 6),
        "path_valid_rate": round(sum(r["path_valid"] for r in records) / len(records), 6),
        "graph_answerability": {label: sum(r["graph_answerability"] == label for r in records)
                                for label in ("full", "partial", "none")},
        "strata": {name: aggregate(values) for name, values in sorted(strata.items())}, "records": records}
    run_id = f"primekgqa_{args.split}_{args.variant}_patterns_{args.seed}_{len(records)}"
    artifact = ROOT / "artifacts/experiments/primekgqa" / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "records"} | {"run_id": run_id}
    (ROOT / f"data/manifests/{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
