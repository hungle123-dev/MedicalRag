from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.dependencies import get_trace_store

router = APIRouter(tags=["traces"])


@router.get("/v1/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, object]:
    value = get_trace_store().get(trace_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return value
