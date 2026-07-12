"""Retrieval quality, scored INDEPENDENTLY of answer accuracy — a question's
gold_terms identify correct evidence; an item is a hit if any gold term
appears in its content. This separates "found the right thing" from
"answered correctly" (design doc: they can diverge)."""
from medgraphrag.core.types import Question, Prediction


def _is_hit(content: str, gold_terms: tuple[str, ...]) -> bool:
    c = content.lower()
    return any(g.lower() in c for g in gold_terms)


def retrieval_recall(preds: list[Prediction], questions: list[Question]) -> float:
    gold = {q.qid: q.gold_terms for q in questions}
    scored = [p for p in preds if gold.get(p.qid)]
    if not scored:
        return 0.0
    hits = sum(1 for p in scored if any(_is_hit(e.content, gold[p.qid]) for e in p.evidence))
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
