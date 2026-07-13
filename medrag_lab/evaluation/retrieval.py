from __future__ import annotations

import math
from collections.abc import Sequence


def unique_ranked(values: Sequence[str], k: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in map(str, values):
        if value not in seen:
            seen.add(value)
            result.append(value)
            if len(result) == k:
                break
    return result


def retrieval_metrics(ranked: Sequence[str], gold: set[str], k: int = 10) -> dict[str, float]:
    ranked_k = unique_ranked(ranked, k)
    gold = set(map(str, gold))
    if not gold:
        return {"ap": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "hit": 0.0}
    hits = 0
    precision_sum = reciprocal_rank = dcg = 0.0
    for rank, value in enumerate(ranked_k, 1):
        if value in gold:
            hits += 1
            precision_sum += hits / rank
            reciprocal_rank = reciprocal_rank or 1.0 / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "ap": precision_sum / ideal_hits,
        "recall": hits / len(gold),
        "mrr": reciprocal_rank,
        "ndcg": dcg / idcg if idcg else 0.0,
        "hit": float(hits > 0),
    }
