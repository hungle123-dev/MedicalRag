import re

_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(tok: str) -> str:
    # Light suffix strip so "lowers"/"lowered"/"lowering" collapse.
    # ponytail: crude by design; swap for a real stemmer if recall needs it.
    for suf in _SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())]
