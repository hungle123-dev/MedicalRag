import os

import openai
import pytest

from medgraphrag.llm.openai_client import OpenAILLM


def test_no_key_anywhere_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAILLM(api_key="")


_has_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_key, reason="no OPENAI_API_KEY set")
def test_real_openai_answers_a_simple_mcqa():
    llm = OpenAILLM()
    try:
        choice = llm.choose(
            "What is the powerhouse of the cell?",
            {"A": "nucleus", "B": "mitochondria", "C": "ribosome", "D": "golgi"},
            context="",
        )
    except openai.AuthenticationError:
        pytest.skip("OPENAI_API_KEY is set but rejected by the API (invalid/expired key)")
    except openai.APIConnectionError as e:
        pytest.skip(f"network cannot reach OpenAI API from this environment: {e}")
    assert choice == "B"
