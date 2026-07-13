from __future__ import annotations

from medrag_lab.schemas import RetrievedDocument


def reciprocal_rank_fusion(
    sparse: list[RetrievedDocument], dense: list[RetrievedDocument], k: int = 60
) -> list[RetrievedDocument]:
    rows: dict[str, RetrievedDocument] = {}
    scores: dict[str, float] = {}
    for ranking in (sparse, dense):
        for rank, row in enumerate(ranking, 1):
            rows.setdefault(row.pmid, row)
            scores[row.pmid] = scores.get(row.pmid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda pmid: (-scores[pmid], pmid))
    return [
        rows[pmid].model_copy(update={"score": scores[pmid], "rank": rank, "retriever": "rrf"})
        for rank, pmid in enumerate(ordered, 1)
    ]
