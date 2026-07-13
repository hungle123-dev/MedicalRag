from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence


def nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    ordered = sorted(map(float, values))
    return ordered[math.ceil(probability * len(ordered)) - 1]


def paired_mde_80(
    left: Sequence[float], right: Sequence[float], groups: Sequence[str], alpha: float = 0.05
) -> float:
    """Normal-approximation two-sided MDE using independent normalized-question groups."""
    if not left or not (len(left) == len(right) == len(groups)):
        raise ValueError("left, right and groups must be non-empty and equally sized")
    from scipy.stats import norm

    grouped: dict[str, list[float]] = defaultdict(list)
    for left_value, right_value, group in zip(left, right, groups, strict=True):
        grouped[str(group)].append(right_value - left_value)
    group_differences = [statistics.fmean(values) for values in grouped.values()]
    if len(group_differences) < 2:
        return 0.0
    critical = norm.ppf(1 - alpha / 2) + norm.ppf(0.8)
    return float(critical * statistics.stdev(group_differences) / math.sqrt(len(group_differences)))


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


def paired_effect_size(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("left and right must be non-empty and equally sized")
    differences = [b - a for a, b in zip(left, right, strict=True)]
    deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    return statistics.fmean(differences) / deviation if deviation else 0.0


def paired_permutation_p(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260713,
) -> float:
    if not left or len(left) != len(right):
        raise ValueError("left and right must be non-empty and equally sized")
    differences = [b - a for a, b in zip(left, right, strict=True)]
    observed = abs(statistics.fmean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        permuted = statistics.fmean(
            value if rng.getrandbits(1) else -value for value in differences
        )
        extreme += abs(permuted) >= observed
    return (extreme + 1) / (resamples + 1)
