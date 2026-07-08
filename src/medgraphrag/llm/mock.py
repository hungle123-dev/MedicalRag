class MockLLM:
    """Deterministic stand-in LLMClient for tests / no-key runs.

    Picks the option whose text appears verbatim in the retrieved context;
    else a rule keyed on question substring; else `default`. Lets tests prove
    "retrieval changed the answer" without a real model.
    """

    def __init__(self, rules: dict[str, str] | None = None, default: str = "A"):
        self._rules = rules or {}
        self._default = default

    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
        # Context is ranked (top-scored item first); reward the option whose
        # text appears EARLIEST, so higher-ranked evidence wins when several
        # option-texts co-occur.
        ctx = context.lower()
        best_key, best_pos = None, None
        for key, text in options.items():
            pos = ctx.find(text.lower())
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_key, best_pos = key, pos
        if best_key is not None:
            return best_key
        for needle, key in self._rules.items():
            if needle.lower() in question_text.lower():
                return key
        return self._default
