import os

import openai
import pytest

from medgraphrag.config import load_env
from medgraphrag.llm.openai_client import OpenAICompatLLM, _extract_letter, _format_options

load_env()  # pull .env -> os.environ for the real-call test


def test_extract_letter_valid():
    assert _extract_letter("The answer is C.", {"A", "B", "C", "D"}) == "C"


def test_extract_letter_rejects_out_of_range():
    assert _extract_letter("Z", {"A", "B"}) is None


def test_format_options_sorted():
    assert _format_options({"B": "y", "A": "x"}) == "A. x\nB. y"


def test_no_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatLLM("gpt-4.1-nano", api_key="")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
@pytest.mark.parametrize("model", ["gpt-4.1-nano", "gemini-2.5-flash-lite"])
def test_real_proxy_answers_simple_mcqa(model):
    llm = OpenAICompatLLM(model)
    try:
        choice = llm.choose(
            "What is the powerhouse of the cell?",
            {"A": "nucleus", "B": "mitochondria", "C": "ribosome", "D": "golgi"},
            context="",
        )
    except (openai.APIConnectionError, openai.InternalServerError) as e:
        pytest.skip(f"proxy unreachable/model unavailable: {e}")
    assert choice == "B"
