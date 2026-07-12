"""Arm = retriever + LLM composed into answer(question) -> Prediction.
Registers E0 (no retrieval) / E1 (BM25) via medgraphrag.core.registry so a
config file selects an arm by name; a new arm (E2/E3/...) is a new module
with its own @registry.retriever(...), never an edit here.
"""
from medgraphrag.core.types import Question, Prediction
from medgraphrag.core.protocols import Retriever, LLMClient
from medgraphrag.core.registry import retriever, build
from medgraphrag.retrieval.null import NullRetriever
from medgraphrag.retrieval.bm25 import BM25Retriever


class Arm:
    def __init__(self, retriever_: Retriever, llm: LLMClient, k: int = 5):
        self._retriever = retriever_
        self._llm = llm
        self._k = k

    def answer(self, q: Question) -> Prediction:
        items = self._retriever.retrieve(q.text, self._k)
        context = "\n".join(i.content for i in items)
        choice = self._llm.choose(q.text, q.options, context)
        return Prediction(qid=q.qid, choice=choice, evidence=tuple(items))


@retriever("E0")
def _build_e0(ctx):
    return NullRetriever()


@retriever("E1")
def _build_e1(ctx):
    return BM25Retriever(ctx["corpus"])


def build_arm(name: str, corpus: dict[str, str], llm: LLMClient, k: int = 5) -> Arm:
    r = build("retriever", name, {"corpus": corpus})
    return Arm(r, llm, k)
