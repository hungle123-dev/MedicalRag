from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QuestionType = Literal["yesno", "factoid", "list", "summary"]


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2_000)
    pipeline_id: str = Field(default="bm25_rag", pattern=r"^[a-z0-9_-]{1,64}$")

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        return value.strip()


class Citation(BaseModel):
    pmid: str = Field(pattern=r"^\d+$")
    claim_ids: list[str] = Field(default_factory=list)
    title: str = ""
    snippet: str = ""
    url: str = ""


class AnswerResponse(BaseModel):
    predicted_type: QuestionType
    exact_answer: str | list[str] | None
    ideal_answer: str = Field(max_length=4_000)
    citations: list[Citation]
    abstained: bool = False
    evidence_support_score: float | None = Field(default=None, ge=0, le=1)
    trace_id: str
    pipeline_id: str
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class RetrievedDocument(BaseModel):
    pmid: str
    title: str
    text: str
    url: str
    score: float
    rank: int = Field(ge=1)
    retriever: str
