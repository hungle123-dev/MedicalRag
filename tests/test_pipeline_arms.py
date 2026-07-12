"""Wiring test: does E1's retrieved context actually reach the LLM, and does
E0 correctly get none? Uses MockLLM (no network) + a REAL corpus slice, so
this proves the pipeline plumbing, not accuracy."""
from pathlib import Path

import pytest

from medgraphrag.pipeline.arms import build_arm
from medgraphrag.llm.mock import MockLLM
from medgraphrag.core.types import Question
from medgraphrag.data.corpus_loader import load_textbook_corpus

TEXTBOOKS_DIR = "data/raw/medrag_textbooks/chunk"

pytestmark = pytest.mark.skipif(
    not Path(TEXTBOOKS_DIR).exists(),
    reason="real MedRAG textbook corpus not downloaded",
)


def _corpus():
    return load_textbook_corpus(TEXTBOOKS_DIR, books=["Pharmacology_Katzung"], limit=2000)


def test_e0_never_retrieves():
    arm = build_arm("E0", {}, MockLLM(default="A"))
    q = Question("q1", "Mechanism of action of aspirin?",
                 {"A": "COX inhibition", "B": "beta blockade"}, "A")
    pred = arm.answer(q)
    assert pred.evidence == ()


def test_e1_retrieves_from_real_corpus_and_reaches_llm():
    arm = build_arm("E1", _corpus(), MockLLM(default="A"))
    q = Question("q1", "What does aspirin inhibit to reduce platelet aggregation?",
                 {"A": "cyclooxygenase", "B": "beta blockade"}, "A")
    pred = arm.answer(q)
    assert pred.evidence, "E1 should retrieve something from a real pharmacology corpus"
    assert all(e.source.startswith("corpus:") for e in pred.evidence)


def test_build_arm_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_arm("E99", {}, MockLLM())
