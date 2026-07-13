from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.dependencies import get_pipeline
from medrag_lab.schemas import AnswerRequest, AnswerResponse

router = APIRouter(tags=["answer"])


@router.post("/v1/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    try:
        return get_pipeline(request.pipeline_id).answer(request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Upstream generation failed") from exc
