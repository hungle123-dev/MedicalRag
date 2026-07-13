"""Pre-registered paired statistics for BioASQ and reviewer calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_bootstrap(left: list[float], right: list[float], seed: int = 20260712, resamples: int = 10_000) -> dict:
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    if not len(delta) or len(left) != len(right): raise ValueError("Paired arrays must have equal non-zero length")
    rng = np.random.default_rng(seed)
    distribution = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    return {"mean_delta_right_minus_left": float(delta.mean()),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(), "resamples": resamples}


def weighted_kappa(left: list[int], right: list[int], categories: int = 3) -> float:
    if len(left) != len(right) or not left: raise ValueError("Reviewer arrays must be paired")
    matrix = np.zeros((categories, categories), dtype=float)
    for a, b in zip(left, right): matrix[a, b] += 1
    observed = matrix / matrix.sum()
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / matrix.sum() ** 2
    weights = np.fromfunction(lambda i, j: ((i - j) / (categories - 1)) ** 2, (categories, categories))
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path, help="JSONL with b3_correctness, g2_correctness, reviewer_a, reviewer_b")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"b3_vs_g2": paired_bootstrap([r["b3_correctness"] for r in rows], [r["g2_correctness"] for r in rows]),
              "reviewer_weighted_kappa": weighted_kappa([r["reviewer_a"] for r in rows], [r["reviewer_b"] for r in rows])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
