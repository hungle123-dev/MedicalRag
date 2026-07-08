"""Retrieval quality, scored INDEPENDENTLY of answer accuracy.

A question's gold_terms identify the correct evidence. An evidence item is a
"hit" if any gold term appears in its content. This separates "did the
retriever find the right thing" from "did the LLM answer correctly" — a
retriever can win one and lose the other (design doc mục 3).
"""
from medgraphrag.core.types import Question, Prediction


def _is_hit(content: str, gold_terms: tuple[str, ...]) -> bool:
    c = content.lower()
    return any(g.lower() in c for g in gold_terms)


def recall_at_k(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.gold_terms for q in questions}
    scored = [p for p in preds if gold.get(p.qid)]
    if not scored:
        return 0.0
    hits = 0
    for p in scored:
        if any(_is_hit(e.content, gold[p.qid]) for e in p.evidence):
            hits += 1
    return hits / len(scored)


def mrr(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.gold_terms for q in questions}
    scored = [p for p in preds if gold.get(p.qid)]
    if not scored:
        return 0.0
    total = 0.0
    for p in scored:
        for rank, e in enumerate(p.evidence, start=1):
            if _is_hit(e.content, gold[p.qid]):
                total += 1.0 / rank
                break
    return total / len(scored)
