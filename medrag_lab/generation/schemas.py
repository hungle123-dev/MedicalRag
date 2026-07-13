from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from medrag_lab.schemas import QuestionType


class GeneratedAnswer(BaseModel):
    predicted_type: QuestionType
    exact_answer: str | list[str] | None
    ideal_answer: str = Field(min_length=1, max_length=4_000)
    citation_pmids: list[str] = Field(default_factory=list)
    abstained: bool = False
    evidence_support_score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("citation_pmids")
    @classmethod
    def valid_pmids(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(map(str, values)))
        if any(not value.isdigit() for value in cleaned):
            raise ValueError("citation_pmids must contain digits only")
        return cleaned


class GatewayResult(BaseModel):
    answer: GeneratedAnswer
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float
    cached: bool = False
    attempts: int = Field(default=1, ge=1)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    raw_response: str = ""
