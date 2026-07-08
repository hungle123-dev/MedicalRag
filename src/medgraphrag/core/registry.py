"""Name -> arm-builder registry.

A new strategy registers itself with @register("E3") and becomes runnable
from a YAML config without editing any existing file. Builders take a
resolved config dict and return an object satisfying the Arm protocol.
"""
from typing import Callable

_ARMS: dict[str, Callable] = {}


def register(name: str):
    def deco(builder: Callable):
        if name in _ARMS:
            raise ValueError(f"arm already registered: {name}")
        _ARMS[name] = builder
        return builder
    return deco


def build(name: str, ctx: dict):
    if name not in _ARMS:
        raise ValueError(f"unknown arm: {name!r} (registered: {sorted(_ARMS)})")
    return _ARMS[name](ctx)


def registered_names() -> list[str]:
    return sorted(_ARMS)
