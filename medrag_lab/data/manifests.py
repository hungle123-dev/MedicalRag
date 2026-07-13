from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PINNED_REVISION = "6d6add1a6ec2090991386b5ae7608b71fd637bc4"
DATASET_URL = "https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag"
LICENSE = "CC BY-NC-SA 4.0"
EXPECTED = {
    "corpus.jsonl": {
        "rows": 49_513,
        "bytes": 126_455_681,
        "sha256": "1e992bd761413d17b3fbb410a368532227d4bc28d0ef27817dde98ef3ecb2ca0",
    },
    "dev.jsonl": {
        "rows": 5_049,
        "bytes": 25_960_520,
        "sha256": "19378f4c5eb4957753bdd7ce67fc570d6d709b6b738b9e0d5ca179e354e0510d",
    },
    "eval.jsonl": {
        "rows": 340,
        "bytes": 4_436_186,
        "sha256": "61a1521015fc190dcc8c8c0f0d1b3a25ea6a694f915461256557a81c4d4bfbdf",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_file(path: Path, row_count: int) -> dict[str, Any]:
    expected = EXPECTED[path.name]
    actual = {"rows": row_count, "bytes": path.stat().st_size, "sha256": sha256(path)}
    actual["matches_pin"] = all(actual[key] == expected[key] for key in expected)
    if not actual["matches_pin"]:
        raise ValueError(f"Pinned file mismatch: {path.name}: {actual}")
    return actual
