import random


def one_hop(paths: list[dict]) -> list[dict]:
    return [path for path in paths if path.get("hop_count") == 1]


def no_path_reranker(paths: list[dict], k: int = 5) -> list[dict]:
    return sorted(paths, key=lambda path: path["id"])[:k]


def matched_random_paths(candidates: list[dict], target: list[dict], seed: int) -> list[dict]:
    """Sample non-target paths matched on hop count and nearest verbalized length."""
    rng, selected = random.Random(seed), []
    for path in target:
        pool = [candidate for candidate in candidates
                if candidate.get("hop_count") == path.get("hop_count")
                and candidate["id"] not in {item["id"] for item in target + selected}]
        if not pool:
            continue
        target_words = len(path.get("snippet", "").split())
        distance = min(abs(len(item.get("snippet", "").split()) - target_words) for item in pool)
        nearest = sorted((item for item in pool
                          if abs(len(item.get("snippet", "").split()) - target_words) == distance),
                         key=lambda item: item["id"])
        selected.append(rng.choice(nearest) | {"matched_target_id": path["id"]})
    return selected


def matched_extra_text(candidates: list[dict], target_graphs: list[dict]) -> list[dict]:
    """Replace every graph item with one text item trimmed to the same word count."""
    selected = []
    for item, target in zip(candidates, target_graphs):
        words = item.get("snippet", "").split()
        limit = len(target.get("snippet", "").split())
        if limit:
            selected.append(item | {"snippet": " ".join(words[:limit]),
                                    "matched_target_id": target["id"]})
    return selected
