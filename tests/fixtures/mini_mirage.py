# Re-export the in-src demo dataset so tests and runner share one source.
from medgraphrag.data.fixture_dataset import QUESTIONS, CORPUS, TRIPLES

__all__ = ["QUESTIONS", "CORPUS", "TRIPLES"]
