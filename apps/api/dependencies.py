from __future__ import annotations

from functools import lru_cache

from medrag_lab.pipeline import MedicalRAGPipeline
from medrag_lab.tracking.traces import TraceStore


@lru_cache(maxsize=8)
def get_pipeline(pipeline_id: str) -> MedicalRAGPipeline:
    return MedicalRAGPipeline(pipeline_id)


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    return TraceStore()
