from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import atomic_json, sha256, stable_hash
from medrag_lab.settings import ROOT, settings

WORD = re.compile(r"[\w]+(?:[-_/][\w]+)*", re.UNICODE)
TYPES = ("yesno", "factoid", "list", "summary")


def normalize_question(text: str) -> str:
    return " ".join(WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def _order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _sample(rows: list[dict[str, Any]], per_type: int, seed: int) -> list[str]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["type"])].append(row)
    selected: list[str] = []
    for kind in TYPES:
        candidates = sorted(by_type[kind], key=lambda row: _order(str(row["question_id"]), seed))
        if len(candidates) < per_type:
            raise ValueError(f"Not enough {kind} rows for {per_type}/type")
        selected.extend(str(row["question_id"]) for row in candidates[:per_type])
    return sorted(selected)


def freeze_splits(seed: int = 20260713) -> dict[str, Any]:
    data_dir = settings().medrag_data_dir
    dev_path, eval_path = data_dir / "dev.jsonl", data_dir / "eval.jsonl"
    dev_rows, eval_rows = list(iter_jsonl(dev_path)), list(iter_jsonl(eval_path))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        groups[normalize_question(str(row["question"]))].append(row)

    validation_ids: set[str] = set()
    for kind in TYPES:
        singletons = [
            rows[0] for rows in groups.values() if len(rows) == 1 and str(rows[0]["type"]) == kind
        ]
        singletons.sort(key=lambda row: _order(str(row["question_id"]), seed + 1))
        if len(singletons) < 50:
            raise ValueError(f"Cannot freeze exact group-safe validation sample for {kind}")
        validation_ids.update(str(row["question_id"]) for row in singletons[:50])

    selection = [row for row in dev_rows if str(row["question_id"]) not in validation_ids]
    result: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "policy": "normalized-question-group-safe; stratified; SHA256 stable ordering",
        "selection4849": sorted(str(row["question_id"]) for row in selection),
        "validation200": sorted(validation_ids),
        "smoke40": _sample(selection, 10, seed + 2),
        "query800": _sample(selection, 200, seed + 3),
        "generation160": _sample(selection, 40, seed + 4),
        "heldout340": sorted(str(row["question_id"]) for row in eval_rows),
        "judge160": _sample(eval_rows, 40, seed + 5),
        "raw_hashes": {"dev": sha256(dev_path), "eval": sha256(eval_path)},
    }
    result["freeze_hash"] = stable_hash(result)
    path = ROOT / "data" / "manifests" / "splits.json"
    atomic_json(path, result)
    verify_splits(path)
    return result


def verify_splits(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "data" / "manifests" / "splits.json"
    splits = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "selection4849": 4_849,
        "validation200": 200,
        "smoke40": 40,
        "query800": 800,
        "generation160": 160,
        "heldout340": 340,
        "judge160": 160,
    }
    for name, count in expected.items():
        if len(splits[name]) != count or len(set(splits[name])) != count:
            raise ValueError(f"{name}: expected {count} unique IDs")
    selection, validation = set(splits["selection4849"]), set(splits["validation200"])
    heldout = set(splits["heldout340"])
    if selection & validation or (selection | validation) & heldout:
        raise ValueError("Population ID overlap")
    for child in ("smoke40", "query800", "generation160"):
        if not set(splits[child]) <= selection:
            raise ValueError(f"{child} is not a selection subset")
    if not set(splits["judge160"]) <= heldout:
        raise ValueError("judge160 is not a heldout subset")

    populations: dict[str, set[str]] = defaultdict(set)
    dev_path = settings().medrag_data_dir / "dev.jsonl"
    for row in iter_jsonl(dev_path):
        question_id = str(row["question_id"])
        population = "selection" if question_id in selection else "validation"
        populations[normalize_question(str(row["question"]))].add(population)
    crossings = sum(len(population) > 1 for population in populations.values())
    if crossings:
        raise ValueError(f"Normalized duplicate groups cross splits: {crossings}")
    expected_hashes = splits["raw_hashes"]
    eval_path = settings().medrag_data_dir / "eval.jsonl"
    if expected_hashes != {"dev": sha256(dev_path), "eval": sha256(eval_path)}:
        raise ValueError("Raw data changed after split freeze")
    return {"status": "ok", "counts": expected, "duplicate_group_crossings": 0}
