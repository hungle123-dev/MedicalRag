from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence


def paired_group_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    groups: Sequence[str],
    *,
    resamples: int = 10_000,
    seed: int = 20260713,
) -> dict[str, float]:
    if not left or not (len(left) == len(right) == len(groups)):
        raise ValueError("left, right and groups must be non-empty and equally sized")
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_rows[str(group)].append(index)
    names = sorted(group_rows)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(names) for _ in names]
        indexes = [index for group in sampled for index in group_rows[group]]
        deltas.append(statistics.fmean(right[index] - left[index] for index in indexes))
    deltas.sort()
    return {
        "mean_delta_right_minus_left": statistics.fmean(
            right_value - left_value for left_value, right_value in zip(left, right, strict=True)
        ),
        "ci95_low": deltas[math.floor(0.025 * (len(deltas) - 1))],
        "ci95_high": deltas[math.ceil(0.975 * (len(deltas) - 1))],
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    for rank, (original_index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[original_index] = running
    return adjusted
