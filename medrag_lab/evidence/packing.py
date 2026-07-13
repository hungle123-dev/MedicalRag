from __future__ import annotations

import tiktoken

from medrag_lab.evidence.snippets import Snippet

ENCODING = tiktoken.get_encoding("o200k_base")


def strongest_in_middle(snippets: list[Snippet]) -> list[Snippet]:
    """Diagnostic ordering: place the highest-ranked item at the context midpoint."""
    if len(snippets) < 3:
        return list(snippets)
    strongest, rest = snippets[0], snippets[1:]
    middle = len(rest) // 2
    return rest[:middle] + [strongest] + rest[middle:]


def source_diverse(snippets: list[Snippet], per_pmid: int = 1) -> list[Snippet]:
    if per_pmid < 1:
        raise ValueError("per_pmid must be positive")
    counts: dict[str, int] = {}
    selected = []
    for snippet in snippets:
        if counts.get(snippet.pmid, 0) < per_pmid:
            selected.append(snippet)
            counts[snippet.pmid] = counts.get(snippet.pmid, 0) + 1
    return selected


def pack_context(snippets: list[Snippet], token_budget: int = 1_200) -> tuple[str, list[Snippet]]:
    if token_budget < 100:
        raise ValueError("Context budget must be at least 100 reference tokens")
    selected: list[Snippet] = []
    blocks: list[str] = []
    used = 0
    for snippet in snippets:
        block = f"[PMID:{snippet.pmid}] {snippet.title}\n{snippet.text}"
        tokens = len(ENCODING.encode(block))
        if used + tokens > token_budget:
            continue
        selected.append(snippet)
        blocks.append(block)
        used += tokens
    return "\n\n".join(blocks), selected


def serialize_context(snippets: list[Snippet]) -> str:
    return "\n\n".join(
        f"[PMID:{snippet.pmid}] {snippet.title}\n{snippet.text}" for snippet in snippets
    )
