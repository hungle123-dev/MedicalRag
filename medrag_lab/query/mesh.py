from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from medrag_lab.data.loaders import iter_jsonl

TOKEN = re.compile(r"[A-Za-z0-9]+")


class MeshExpander:
    """Conservative local MeSH phrase expansion; never reads question gold fields."""

    def __init__(self, corpus: Path):
        counts: Counter[str] = Counter()
        for row in iter_jsonl(corpus):
            counts.update(
                str(term).strip() for term in row.get("mesh_terms", []) if str(term).strip()
            )
        self.lookup: dict[str, tuple[str, int]] = {}
        for term, count in counts.items():
            normalized = " ".join(TOKEN.findall(term.casefold()))
            previous = self.lookup.get(normalized)
            if normalized and (previous is None or count > previous[1]):
                self.lookup[normalized] = (term, count)
        self.max_words = max((len(value.split()) for value in self.lookup), default=1)

    def expand(self, question: str, limit: int = 3) -> list[str]:
        words = TOKEN.findall(question.casefold())
        matches: dict[str, tuple[str, int]] = {}
        for size in range(1, min(self.max_words, len(words)) + 1):
            for start in range(len(words) - size + 1):
                normalized = " ".join(words[start : start + size])
                if normalized in self.lookup:
                    matches[normalized] = self.lookup[normalized]
        additions = [
            value[0]
            for _, value in sorted(
                matches.items(), key=lambda item: (-len(item[0].split()), -item[1][1], item[0])
            )[:limit]
        ]
        return [question] if not additions else [question + " " + " ".join(additions)]
