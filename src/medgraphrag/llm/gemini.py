"""Real LLMClient backed by Gemini (google-genai SDK). Needs GEMINI_API_KEY env var.

choose() sends the question+options+context as a single MCQA prompt and
parses the returned option letter. ponytail: single retry on unparsable
output, no exponential backoff — good enough for a few hundred calls.
"""
import os
import re

from google import genai

_PROMPT = """You are a medical expert answering a multiple-choice question.
{context_block}
Question: {question}
Options:
{options_block}

Reply with ONLY the single letter of the correct option (e.g. "A"). No explanation."""


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def _extract_letter(text: str, valid_keys: set[str]) -> str | None:
    match = re.search(r"\b([A-Z])\b", text.strip().upper())
    if match and match.group(1) in valid_keys:
        return match.group(1)
    return None


class GeminiLLM:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=key)
        self._model = model

    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        context_block = f"Context:\n{context}\n" if context else ""
        prompt = _PROMPT.format(
            context_block=context_block,
            question=question_text,
            options_block=_format_options(options),
        )
        valid = set(options.keys())
        letter = self._ask(prompt, valid)
        if letter is None:
            # ponytail: one retry with a stricter instruction, then give up
            # and fall back to the first option key rather than crashing a run.
            letter = self._ask(prompt + "\n\nReply with EXACTLY one letter.", valid)
        return letter or sorted(valid)[0]

    def _ask(self, prompt: str, valid: set[str]) -> str | None:
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return _extract_letter(response.text or "", valid)
