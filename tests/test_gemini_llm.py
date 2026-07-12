import os

import pytest

from medgraphrag.llm.gemini import _extract_letter, _format_options, GeminiLLM


def test_extract_letter_finds_valid_key():
    assert _extract_letter("The answer is B.", {"A", "B", "C", "D"}) == "B"


def test_extract_letter_rejects_key_not_in_options():
    assert _extract_letter("Z", {"A", "B"}) is None


def test_extract_letter_handles_bare_letter():
    assert _extract_letter("A", {"A", "B", "C"}) == "A"


def test_format_options_sorted():
    assert _format_options({"B": "y", "A": "x"}) == "A. x\nB. y"


def test_missing_api_key_raises():
    with pytest.raises(RuntimeError):
        GeminiLLM(api_key="")


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY set")
def test_real_gemini_answers_a_simple_mcqa():
    llm = GeminiLLM()
    choice = llm.choose(
        "What is the powerhouse of the cell?",
        {"A": "nucleus", "B": "mitochondria", "C": "ribosome", "D": "golgi"},
        context="",
    )
    assert choice == "B"
