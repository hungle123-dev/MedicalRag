from __future__ import annotations

from fastapi import APIRouter

from medrag_lab.experiments.registry import load_registry, validate_registry

router = APIRouter(tags=["experiments"])


@router.get("/v1/experiments")
def experiments() -> dict[str, object]:
    registry = load_registry()
    return {
        "validation": validate_registry(registry),
        "registry_hash": registry["registry_hash"],
        "arms": registry["resolved_arms"],
    }
