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
    registry["registry_hash"] = stable_hash(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> dict[str, int]:
    families = registry.get("families")
    if not isinstance(families, list) or len(families) != 12:
        raise ValueError("Registry must contain exactly 12 families")
    arm_ids: set[str] = set()
    counts = {"core": 0, "stretch": 0}
    for family in families:
        missing = REQUIRED - set(family)
        if missing:
            raise ValueError(f"{family.get('id', '?')}: missing {sorted(missing)}")
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
    return counts
