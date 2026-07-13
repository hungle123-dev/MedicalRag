from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medrag_lab.generation.gateway import GatewayClient
from medrag_lab.schemas import RetrievedDocument


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 100) -> tuple[list[RetrievedDocument], float]: ...


@dataclass(frozen=True)
class IterativeQuery:
    first_query: str
    second_query: str
    seed_pmids: list[str]


class TwoRoundExpander:
    def __init__(self, retriever: Retriever, gateway: GatewayClient, model: str | None = None):
        self.retriever = retriever
        self.gateway = gateway
        self.model = model

    def expand(self, question: str, seed_k: int = 5) -> IterativeQuery:
        documents, _ = self.retriever.retrieve(question, seed_k)
        evidence = "\n".join(f"{item.title}: {item.text[:500]}" for item in documents)
        system = """Rewrite the biomedical search query once using useful terminology found in
the seed abstracts. Do not answer the question and do not add unsupported entities. Return the
standard JSON schema with predicted_type=summary, exact_answer=null, the rewritten query only in
ideal_answer, no citations, abstained=false, evidence_support_score=null."""
        result = self.gateway.generate(
            system_prompt=system,
            user_prompt=f"QUESTION\n{question}\n\nSEED ABSTRACTS\n{evidence}",
            model=self.model,
            max_output_tokens=160,
        )
        return IterativeQuery(
            question, result.answer.ideal_answer, [item.pmid for item in documents]
        )
