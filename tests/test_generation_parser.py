from __future__ import annotations

import pytest
from pydantic import ValidationError

from medrag_lab.generation.parser import parse_generated_answer


def test_parser_normalizes_explicit_abstention_with_empty_answer() -> None:
    answer = parse_generated_answer(
        '{"predicted_type":"summary","exact_answer":null,"ideal_answer":null,'
        '"citation_pmids":[],"abstained":true,"evidence_support_score":0}'
    )
    assert answer.ideal_answer == "Insufficient evidence in the provided context."
    assert answer.abstained is True


def test_parser_rejects_empty_answer_without_abstention() -> None:
    with pytest.raises(ValidationError):
        parse_generated_answer(
            '{"predicted_type":"summary","exact_answer":null,"ideal_answer":null,'
            '"citation_pmids":[],"abstained":false,"evidence_support_score":0}'
        )
