import random


def one_hop(paths: list[dict]) -> list[dict]:
    return [path for path in paths if path.get("hop_count") == 1]


def no_path_reranker(paths: list[dict], k: int = 5) -> list[dict]:
    return sorted(paths, key=lambda path: path["id"])[:k]


def matched_random_paths(candidates: list[dict], target: list[dict], seed: int) -> list[dict]:
    """Sample controls with the same hop counts; degree matching needs a future degree table."""
    # ponytail: hop matching is available now; add degree bins only if graph benefit survives this control.
    rng, selected = random.Random(seed), []
    for path in target:
        pool = [candidate for candidate in candidates
                if candidate.get("hop_count") == path.get("hop_count") and candidate not in selected]
        if pool: selected.append(rng.choice(pool))
    return selected


def matched_extra_text(candidates: list[dict], graph_tokens: int, max_items: int = 5) -> list[dict]:
    selected, used = [], 0
    for item in candidates:
        if len(selected) == max_items or used >= graph_tokens: break
        selected.append(item); used += len(item.get("snippet", "").split())
    return selected
