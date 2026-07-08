"""Dataset seam. `load_dataset(name)` returns (questions, corpus, triples).

Only the synthetic fixture is wired now; a real MIRAGE loader registers here
later under name "mirage" behind the same signature.
"""
from medgraphrag.core.types import Question


def _fixture():
    from medgraphrag.data.fixture_dataset import QUESTIONS, CORPUS, TRIPLES
    return QUESTIONS, CORPUS, TRIPLES


_LOADERS = {"fixture": _fixture}


def load_dataset(name: str):
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset: {name!r} (available: {sorted(_LOADERS)})")
    return _LOADERS[name]()
