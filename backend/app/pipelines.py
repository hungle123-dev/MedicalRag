from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Pipeline:
    id: str
    name: str
    description: str

    def run(self, question: str) -> dict:
        # ponytail: deterministic placeholder; replace per pipeline when model adapters land.
        return {
            "answer": f"[{self.id}] Experimental answer for: {question}",
            "citations": [],
            "evidence": [],
        }


PIPELINES = {
    pipeline.id: pipeline
    for pipeline in (
        Pipeline("B0", "Closed-book", "Generator without retrieval"),
        Pipeline("B1", "BM25 RAG", "Lexical text retrieval"),
        Pipeline("B2", "MedCPT RAG", "Biomedical dense retrieval"),
        Pipeline("B3", "Hybrid Text RAG", "BM25 + MedCPT + RRF + reranker"),
        Pipeline("G1", "PrimeKG RAG", "Knowledge-graph retrieval only"),
        Pipeline("G2", "Hybrid Text + Graph", "B3 with PrimeKG evidence"),
    )
}


def list_pipelines() -> list[dict]:
    return [asdict(pipeline) for pipeline in PIPELINES.values()]

