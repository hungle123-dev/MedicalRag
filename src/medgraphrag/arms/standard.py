"""Standard arm = retriever + prompt + LLM, and registry entries E0/E1/E2.

`ctx` (from a resolved config) carries: corpus, triples, llm, k. A new
strategy adds its own module with @register(...) and never touches this file.
"""
from medgraphrag.core.types import Question, Prediction
from medgraphrag.core.registry import register
from medgraphrag.retrieval.null import NullRetriever
from medgraphrag.retrieval.text import TextRetriever
from medgraphrag.retrieval.triple import TripleRetriever


class StandardArm:
    def __init__(self, retriever, llm, k: int = 3):
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def answer(self, q: Question) -> Prediction:
        items = self._retriever.retrieve(q.text, self._k)
        context = "\n".join(i.content for i in items)
        choice = self._llm.choose(q.text, q.options, context)
        return Prediction(qid=q.qid, choice=choice, evidence=tuple(items))


@register("E0")
def _build_e0(ctx):
    return StandardArm(NullRetriever(), ctx["llm"], ctx.get("k", 3))


@register("E1")
def _build_e1(ctx):
    return StandardArm(TextRetriever(ctx["corpus"]), ctx["llm"], ctx.get("k", 3))


@register("E2")
def _build_e2(ctx):
    return StandardArm(TripleRetriever(ctx["triples"]), ctx["llm"], ctx.get("k", 3))
