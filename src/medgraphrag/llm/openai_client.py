"""Real LLMClient backed by OpenAI. Needs OPENAI_API_KEY env var.

Mirrors gemini.py's contract exactly (same prompt shape, same parsing) so an
arm config can swap "gemini" <-> "openai" without touching pipeline code.
"""
import os

from openai import OpenAI

from medgraphrag.llm.gemini import _PROMPT, _format_options, _extract_letter


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key)
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
            letter = self._ask(prompt + "\n\nReply with EXACTLY one letter.", valid)
        return letter or sorted(valid)[0]

    def _ask(self, prompt: str, valid: set[str]) -> str | None:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content or ""
        return _extract_letter(text, valid)
