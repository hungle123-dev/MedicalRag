from __future__ import annotations

import re

from medrag_lab.evidence.snippets import Snippet
from medrag_lab.schemas import Citation

PMID = re.compile(r"\[PMID:(\d+)\]")


def citations_from_answer(answer: str, evidence: list[Snippet]) -> list[Citation]:
    by_pmid = {snippet.pmid: snippet for snippet in evidence}
    citations: list[Citation] = []
    for pmid in dict.fromkeys(PMID.findall(answer)):
        if pmid in by_pmid:
            snippet = by_pmid[pmid]
            citations.append(
                Citation(
                    pmid=pmid,
                    title=snippet.title,
                    snippet=snippet.text,
                    url=snippet.url,
                )
            )
    return citations
