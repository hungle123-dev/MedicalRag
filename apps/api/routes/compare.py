from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from apps.api.dependencies import get_pipeline
from medrag_lab.schemas import AnswerRequest, AnswerResponse

router = APIRouter(tags=["compare"])


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2_000)
    pipeline_ids: list[str] = Field(min_length=2, max_length=2)


class CompareResponse(BaseModel):
    answers: list[AnswerResponse]


@router.post("/v1/compare", response_model=CompareResponse)
def compare(request: CompareRequest) -> CompareResponse:
    if len(set(request.pipeline_ids)) != 2:
        raise HTTPException(status_code=422, detail="Select two distinct pipelines")
    try:
        answers = [
            get_pipeline(pipeline_id).answer(
                AnswerRequest(question=request.question, pipeline_id=pipeline_id)
            )
            for pipeline_id in request.pipeline_ids
        ]
        return CompareResponse(answers=answers)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Upstream generation failed") from exc
