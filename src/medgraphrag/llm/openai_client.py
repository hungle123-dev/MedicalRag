"""Single LLMClient over any OpenAI-compatible endpoint (real OpenAI, or the
futureppo proxy that also serves Gemini). base_url + key come from env
(.env -> os.environ). One class runs every model we use.
"""
import os
import re

from openai import OpenAI

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


class OpenAICompatLLM:
    """model is any id the endpoint serves: gpt-4.1-nano, gemini-2.5-flash-lite, ..."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 2048):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = base_url or os.environ.get("OPENAI_BASE_URL")  # None -> real OpenAI
        # hard 30s timeout + no SDK-level retry: a hung call fails fast so our
        # own retry loop (run_experiment) controls backoff instead of the SDK
        # silently blocking for its 600s default.
        self._client = OpenAI(api_key=key, base_url=url, timeout=30.0, max_retries=0)
        self._model = model
        # reasoning models (gemini) return empty with a tiny budget; give headroom
        self._max_tokens = max_tokens

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
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""
        return _extract_letter(text, valid)
