"""String -> builder maps, one per pluggable axis.

A new variant registers with @retriever("E5") and becomes runnable from YAML
without editing existing files. ponytail: plain dict maps, not a plugin
framework — that is all the ablation matrix needs.
"""
from typing import Callable

_REGISTRIES: dict[str, dict[str, Callable]] = {}


def _reg(kind: str):
    table = _REGISTRIES.setdefault(kind, {})

    def register(name: str):
        def deco(fn: Callable):
            if name in table:
                raise ValueError(f"{kind} already registered: {name}")
            table[name] = fn
            return fn
        return deco

    return register


def build(kind: str, name: str, ctx: dict):
    table = _REGISTRIES.get(kind, {})
    if name not in table:
        raise ValueError(f"unknown {kind}: {name!r} (have: {sorted(table)})")
    return table[name](ctx)


def names(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}))


# axis-specific decorators
retriever = _reg("retriever")
graph_retriever = _reg("graph_retriever")
fusion = _reg("fusion")
reranker = _reg("reranker")
query_builder = _reg("query_builder")
answerer = _reg("answerer")
