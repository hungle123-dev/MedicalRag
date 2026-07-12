class MockLLM:
    """Deterministic stand-in LLMClient — no network, no key. Used to test
    pipeline WIRING (does E1's retrieved context reach the LLM correctly),
    never to report accuracy numbers.

    Picks the option whose text appears EARLIEST in the ranked context
    (rewards higher-ranked evidence); falls back to a substring rule, then a
    fixed default.
    """

    def __init__(self, rules: dict[str, str] | None = None, default: str = "A"):
        self._rules = rules or {}
        self._default = default

    def choose(self, question_text: str, options: dict[str, str], context: str) -> str:
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
