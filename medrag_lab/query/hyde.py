from __future__ import annotations

from dataclasses import dataclass

from medrag_lab.generation.gateway import GatewayClient

HYDE_SYSTEM = """Generate one concise hypothetical PubMed-style abstract that would answer the
biomedical question. It is retrieval text, not a factual answer. Return the standard JSON answer
schema with predicted_type=summary, exact_answer=null, ideal_answer containing only the abstract,
citation_pmids=[], abstained=false, and evidence_support_score=null."""


@dataclass(frozen=True)
class HyDEQuery:
    original: str
    hypothetical_document: str

    @property
    def expanded(self) -> str:
        return f"{self.original} {self.hypothetical_document}"


class HyDEExpander:
    def __init__(self, gateway: GatewayClient, model: str | None = None):
        self.gateway = gateway
        self.model = model

    def expand(self, question: str) -> HyDEQuery:
        result = self.gateway.generate(
            system_prompt=HYDE_SYSTEM,
            user_prompt=f"BIOMEDICAL QUESTION\n{question}",
            model=self.model,
            max_output_tokens=300,
        )
        return HyDEQuery(question, result.answer.ideal_answer)
