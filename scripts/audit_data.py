"""Generate reproducible schema, leakage and coverage audits from downloaded data."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/manifests"


def atomic_json(name: str, value) -> None:
    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def normalized_tokens(text: str) -> frozenset[str]:
    return frozenset("".join(ch.casefold() if ch.isalnum() else " " for ch in text).split())


def near_overlap(left: list[dict], right: list[dict], threshold: float = 0.9) -> list[dict]:
    # Small split (5,049 x 340): exact all-pairs Jaccard is deterministic and auditable.
    left_tokens = [(row["question_id"], normalized_tokens(row["question"])) for row in left]
    matches = []
    for row in right:
        tokens = normalized_tokens(row["question"])
        for question_id, candidate in left_tokens:
            union = tokens | candidate
            similarity = len(tokens & candidate) / len(union) if union else 1.0
            if similarity >= threshold:
                matches.append({"dev_id": question_id, "eval_id": row["question_id"],
                                "token_jaccard": round(similarity, 6)})
    return matches


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n)); self.size = [1] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right: return
        if self.size[left] < self.size[right]: left, right = right, left
        self.parent[right] = left; self.size[left] += self.size[right]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_jsonl(RAW / "bioasq/corpus.jsonl")
    dev = load_jsonl(RAW / "bioasq/dev.jsonl")
    evaluation = load_jsonl(RAW / "bioasq/eval.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    lengths = [len(tokenizer(row["title"], row["text"], truncation=False)["input_ids"]) for row in corpus]

    uf = UnionFind(129375)
    with (RAW / "primekg/edges.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            uf.union(int(row["x_index"]), int(row["y_index"]))
    components = Counter(uf.find(index) for index in range(129375))
    component_sizes = sorted(components.values(), reverse=True)

    schema = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bioasq_corpus": {"format": "jsonl", "fields": {key: type(value).__name__ for key, value in corpus[0].items()}},
        "bioasq_questions": {"format": "jsonl", "fields": {key: type(value).__name__ for key, value in dev[0].items()}},
        "primekg_nodes": {"format": "tsv", "fields": ["node_index", "node_id", "node_type", "node_name", "node_source"]},
        "primekg_edges": {"format": "csv", "fields": ["relation", "display_relation", "x_index", "y_index"]},
        "primekgqa": {"format": "json-array", "note": "Full field/type profile is recorded in eda.json."},
    }
    atomic_json("schema.json", schema)

    exact_dev = {" ".join(row["question"].casefold().split()): row["question_id"] for row in dev}
    exact = [{"dev_id": exact_dev[key], "eval_id": row["question_id"]}
             for row in evaluation if (key := " ".join(row["question"].casefold().split())) in exact_dev]
    leakage = {
        "bioasq": {"exact_dev_eval": exact, "near_dev_eval_threshold": 0.9,
                    "near_dev_eval": near_overlap(dev, evaluation)},
        "primekgqa": {"warning": "Synthetic questions overlap substantially across published splits; see eda.json.",
                       "confirmatory_use": "component-only; stratify seen/unseen patterns"},
        "policy": {"inference_index_fields": ["corpus.id", "corpus.title", "corpus.text"],
                   "forbidden_index_fields": ["question", "answer", "relevant_passage_ids", "snippets"]},
    }
    atomic_json("leakage_audit.json", leakage)

    audit = {
        "medcpt_truncation": {"tokenizer": "ncbi/MedCPT-Article-Encoder", "max_length": 512,
            "documents_over_512": sum(length > 512 for length in lengths),
            "rate": round(sum(length > 512 for length in lengths) / len(lengths), 6),
            "max_tokens": max(lengths)},
        "primekg_connectivity": {"connected_components": len(component_sizes),
            "largest_component_nodes": component_sizes[0],
            "largest_component_fraction": round(component_sizes[0] / 129375, 6),
            "isolated_nodes": sum(size == 1 for size in component_sizes)},
        "primekg_edge_provenance": {"row_level_field_available": False,
            "coverage": 0.0, "fallback": "PrimeKG release-level provenance only; do not claim per-edge source."},
    }
    atomic_json("data_quality_audit.json", audit)
    exclusions = OUT / "exclusions.jsonl"
    exclusions.write_text("", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
