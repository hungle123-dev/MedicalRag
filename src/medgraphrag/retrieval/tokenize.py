import re

_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(tok: str) -> str:
    # Light suffix strip so "reduces"/"reduced"/"reducing" collapse to one
    # stem. Crude by design (ponytail: swap for a real stemmer if E3/E4 need
    # recall) — enough to stop trivial surface-form mismatches.
    for suf in _SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]
