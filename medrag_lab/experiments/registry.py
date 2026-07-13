from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from medrag_lab.data.manifests import stable_hash
from medrag_lab.settings import ROOT

REQUIRED = {
    "id",
    "hypothesis",
    "changed_factor",
    "fixed_controls",
    "population",
    "primary_metric",
    "gate",
    "depends_on",
    "arms",
}


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "configs" / "experiments" / "registry.yaml"
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_registry(registry)
    registry["resolved_arms"] = resolve_arms(registry)
    registry["registry_hash"] = stable_hash(registry)
    return registry


def resolve_arms(registry: dict[str, Any]) -> list[dict[str, Any]]:
    resolved = []
    for family in registry["families"]:
        shared = {key: family[key] for key in REQUIRED - {"arms"}}
        for arm in family["arms"]:
            value = shared | arm
            value["config_hash"] = stable_hash(value)
            value["model_hash"] = (
                stable_hash({"alias": arm["variant"]})
                if family["id"] == "E08"
                else "not_applicable"
            )
            value["prompt_hash"] = (
                stable_hash({"prompt_variant": arm["variant"]})
                if family["id"] in {"E08", "E09", "E10", "E11"}
                else "not_applicable"
            )
            resolved.append(value)
    return resolved


def validate_registry(registry: dict[str, Any]) -> dict[str, int]:
    families = registry.get("families")
    if not isinstance(families, list) or len(families) != 12:
        raise ValueError("Registry must contain exactly 12 families")
    family_ids = [family.get("id") for family in families]
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("Family IDs must be unique")
    known_families = set(family_ids)
    arm_ids: set[str] = set()
    counts = {"core": 0, "stretch": 0}
    for family in families:
        missing = REQUIRED - set(family)
        if missing:
            raise ValueError(f"{family.get('id', '?')}: missing {sorted(missing)}")
        unknown = set(family["depends_on"]) - known_families
        if unknown:
            raise ValueError(f"{family['id']}: unknown dependencies {sorted(unknown)}")
        for arm in family["arms"]:
            if set(arm) != {"id", "variant", "tier"}:
                raise ValueError(f"{family['id']}: arm requires id, variant and tier")
            if arm["id"] in arm_ids:
                raise ValueError(f"Duplicate arm ID: {arm['id']}")
            if arm["tier"] not in counts:
                raise ValueError(f"Unknown arm tier: {arm['tier']}")
            arm_ids.add(arm["id"])
            counts[arm["tier"]] += 1
    if counts != {"core": 41, "stretch": 13}:
        raise ValueError(f"Expected 41 core + 13 stretch, got {counts}")

    remaining = {family["id"]: set(family["depends_on"]) for family in families}
    while remaining:
        ready = {family_id for family_id, dependencies in remaining.items() if not dependencies}
        if not ready:
            raise ValueError(f"Dependency cycle among {sorted(remaining)}")
        remaining = {
            family_id: dependencies - ready
            for family_id, dependencies in remaining.items()
            if family_id not in ready
        }
    return counts
