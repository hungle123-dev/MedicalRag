"""Paired bootstrap analysis for cached retrieval experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def paired_bootstrap(left: list[float], right: list[float], seed: int, resamples: int = 10_000) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("Paired samples must have the same non-zero length")
    differences = np.asarray(right) - np.asarray(left)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(differences), size=(resamples, len(differences)))
    distribution = differences[indexes].mean(axis=1)
    return {
        "mean_delta_right_minus_left": round(float(differences.mean()), 6),
        "ci95": [round(float(value), 6) for value in np.quantile(distribution, [0.025, 0.975])],
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    runs = json.loads(args.artifact.read_text(encoding="utf-8"))
    left, right = runs
    if [row["question_id"] for row in left["rows"]] != [row["question_id"] for row in right["rows"]]:
        raise ValueError("Question order differs between paired runs")
    metrics = {
        key: paired_bootstrap(
            [row["metrics"][key] for row in left["rows"]],
            [row["metrics"][key] for row in right["rows"]],
            args.seed,
        )
        for key in ("recall_at_10", "mrr", "ndcg_at_10")
    }
    decision = "C2" if metrics["recall_at_10"]["ci95"][0] > 0 else "C0"
    report = {"left": left["strategy"], "right": right["strategy"], "metrics": metrics, "selected": decision}
    output = ROOT / "data" / "manifests" / "bioasq_chunk_selection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
