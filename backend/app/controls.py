import random


def one_hop(paths: list[dict]) -> list[dict]:
    return [path for path in paths if path.get("hop_count") == 1]


def no_path_reranker(paths: list[dict], k: int = 5) -> list[dict]:
    return sorted(paths, key=lambda path: path["id"])[:k]


def matched_random_paths(candidates: list[dict], target: list[dict], seed: int) -> list[dict]:
    """Sample intact non-target paths matched on hop count and nearest length."""
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
    """Use the earliest unused ranked text long enough for each graph slot."""
    selected, used_indices = [], set()
    for target in target_graphs:
        limit = len(target.get("snippet", "").split())
        match = next(((index, item) for index, item in enumerate(candidates)
                      if index not in used_indices and len(item.get("snippet", "").split()) >= limit), None)
        if limit and match:
            index, item = match
            used_indices.add(index)
            selected.append(item | {"snippet": " ".join(item["snippet"].split()[:limit]),
                                    "matched_target_id": target["id"]})
    return selected


def _coverage(target: list[dict], control: list[dict]) -> tuple[dict, dict, list[str], list[str]]:
    targets = {str(item["id"]): item for item in target}
    seen: dict[str, int] = {}
    unknown_targets = 0
    for item in control:
        matched_id = item.get("matched_target_id")
        if matched_id is None:
            continue
        matched_id = str(matched_id)
        seen[matched_id] = seen.get(matched_id, 0) + 1
        if matched_id not in targets:
            unknown_targets += 1
    missing = sorted(set(targets) - set(seen))
    duplicates = sorted(item_id for item_id, count in seen.items() if count > 1)
    return targets, seen, missing, duplicates


def canonical_path_snippet(item: dict) -> str | None:
    """Reconstruct the only structurally valid verbalization from path metadata."""
    try:
        names = {str(node["id"]): node["name"] for node in item["nodes"]}
        return " · ".join(
            f"{names[str(edge['source_id'])]} —{edge['relation']}→ {names[str(edge['target_id'])]}"
            for edge in item["edges"]
        )
    except (KeyError, TypeError):
        return None


def matched_graph_control_audit(target: list[dict], control: list[dict],
                                forbidden_node_ids: set[str] | None = None) -> dict:
    """Audit post-budget X2 slots, path syntax, hops, and node disjointness."""
    targets, seen, missing, duplicates = _coverage(target, control)
    forbidden = {str(value) for value in (forbidden_node_ids or set())}
    valid = hop_mismatches = structural_invalid = node_overlaps = wrong_type = 0
    unknown_targets = sum(item_id not in targets for item_id in seen)
    for item in control:
        matched_id = str(item.get("matched_target_id"))
        target_item = targets.get(matched_id)
        type_ok = item.get("type") == "graph"
        hop_ok = bool(target_item) and item.get("hop_count") == target_item.get("hop_count")
        structure_ok = canonical_path_snippet(item) == item.get("snippet")
        nodes_ok = not ({str(node.get("id")) for node in item.get("nodes", [])} & forbidden)
        wrong_type += not type_ok
        hop_mismatches += not hop_ok
        structural_invalid += not structure_ok
        node_overlaps += not nodes_ok
        valid += bool(type_ok and hop_ok and structure_ok and nodes_ok)
    complete = (valid == len(targets) == len(control) and not missing and not duplicates and
                not unknown_targets)
    return {
        "requested_slots": len(targets),
        "matched_slots": valid,
        "complete": complete,
        "missing_target_ids": missing,
        "duplicate_target_ids": duplicates,
        "hop_mismatches": hop_mismatches,
        "unknown_target_ids": unknown_targets,
        "structurally_invalid_paths": structural_invalid,
        "forbidden_node_overlaps": node_overlaps,
        "wrong_type_slots": wrong_type,
    }


def matched_text_control_audit(target: list[dict], control: list[dict]) -> dict:
    """Audit post-budget X1 slots and exact per-slot word matching."""
    targets, seen, missing, duplicates = _coverage(target, control)
    valid = length_mismatches = wrong_type = 0
    unknown_targets = sum(item_id not in targets for item_id in seen)
    for item in control:
        target_item = targets.get(str(item.get("matched_target_id")))
        type_ok = item.get("type") == "text"
        length_ok = bool(target_item) and len(item.get("snippet", "").split()) == len(
            target_item.get("snippet", "").split())
        wrong_type += not type_ok
        length_mismatches += not length_ok
        valid += bool(type_ok and length_ok)
    complete = (valid == len(targets) == len(control) and not missing and not duplicates and
                not unknown_targets)
    return {
        "requested_slots": len(targets), "matched_slots": valid, "complete": complete,
        "missing_target_ids": missing, "duplicate_target_ids": duplicates,
        "unknown_target_ids": unknown_targets, "word_length_mismatches": length_mismatches,
        "wrong_type_slots": wrong_type,
    }
