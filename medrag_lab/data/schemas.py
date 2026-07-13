from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medrag_lab.schemas import QuestionType


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    text: str
    url: str
    publication_date: str = ""
    mesh_terms: list[str] = Field(default_factory=list)


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question: str
    answer: str
    relevant_passage_ids: list[str]
    type: QuestionType
    snippets: list[dict[str, Any]]


class InferenceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str = Field(min_length=3, max_length=2_000)
