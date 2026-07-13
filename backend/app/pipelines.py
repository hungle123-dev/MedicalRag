import os
import time
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .retrieval import BM25Index
from .graph import PrimeKGIndex
from .medcpt import MedCPTIndex, MedCPTReranker, reciprocal_rank_fusion
from .generator import create_generator, validate_citations


_bm25: BM25Index | None = None
_graph: PrimeKGIndex | None = None
_medcpt: MedCPTIndex | None = None
_reranker: MedCPTReranker | None = None
_generator = None


def project_root() -> Path:
    return Path(os.getenv("MEDICAL_RAG_ROOT", Path(__file__).parents[2]))


def bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        root = project_root()
        path = Path(os.getenv("MEDICAL_RAG_BM25_INDEX", root / "indexes" / "bm25_c0.pkl"))
        if not path.exists():
            raise RuntimeError(f"BM25 index unavailable: run python scripts/build_indexes.py ({path})")
        _bm25 = BM25Index.load(path)
    return _bm25


def graph() -> PrimeKGIndex:
    global _graph
    if _graph is None:
        path = Path(os.getenv("MEDICAL_RAG_GRAPH_INDEX", project_root() / "indexes" / "primekg.sqlite3"))
        if not path.exists():
            raise RuntimeError(f"PrimeKG index unavailable: run python scripts/build_graph_index.py ({path})")
        _graph = PrimeKGIndex(path)
    return _graph


def medcpt() -> MedCPTIndex:
    global _medcpt
    if _medcpt is None:
        path = Path(os.getenv("MEDICAL_RAG_MEDCPT_INDEX", project_root() / "indexes" / "medcpt"))
        if not (path / "articles.faiss").exists():
            raise RuntimeError(f"MedCPT index unavailable: run python scripts/build_medcpt_index.py ({path})")
        _medcpt = MedCPTIndex(path)
    return _medcpt


def reranker() -> MedCPTReranker:
    global _reranker
    if _reranker is None:
        _reranker = MedCPTReranker()
    return _reranker


def generator():
    global _generator
    if _generator is None:
        _generator = create_generator(project_root())
    return _generator


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def trim_item(item: dict, token_limit: int) -> tuple[dict, int]:
    words = item.get("snippet", "").split()
    used = min(len(words), max(token_limit, 0))
    return item | {"snippet": " ".join(words[:used])}, used


def evidence_budget(texts: list[dict], graphs: list[dict]) -> tuple[list[dict], dict]:
    """Apply the frozen 1,800-token/8-item fusion budget using a logged whitespace proxy."""
    # ponytail: provider tokenizers are unavailable offline; replace proxy when the generator API exposes countTokens.
    selected_graphs: list[dict] = []
    selected_texts: list[dict] = []
    graph_used = text_used = 0
    for item in graphs[:5]:
        if len(selected_graphs) == 8 or graph_used >= 540:
            break
        trimmed, used = trim_item(item, min(540 - graph_used, 180))
        if used:
            selected_graphs.append(trimmed); graph_used += used
    for item in texts:
        if len(selected_graphs) + len(selected_texts) == 8 or graph_used + text_used >= 1800:
            break
        trimmed, used = trim_item(item, 1800 - graph_used - text_used)
        if used:
            selected_texts.append(trimmed); text_used += used
    selected = []
    for index in range(max(len(selected_texts), len(selected_graphs))):
        if index < len(selected_texts): selected.append(selected_texts[index])
        if index < len(selected_graphs): selected.append(selected_graphs[index])
    return selected, {
        "token_count_method": "whitespace_proxy_v1", "token_budget": 1800,
        "ordering": "rank_interleave_text_first_v1",
        "graph_tokens_requested": 540 if graphs else 0, "graph_tokens_actual": graph_used,
        "text_tokens_actual": text_used, "evidence_items": len(selected),
    }


def rows_to_text_evidence(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": f"PMID:{row['id']}", "type": "text", "title": row["title"],
            "snippet": row["text"][:600], "url": row["url"],
            "source": "PubMed", "pmid": str(row["id"]),
            "score": row.get("rerank_score", row.get("rrf_score", row["score"])),
            "retrievers": row.get("retrievers", [row.get("retriever")]),
        }
        for row in rows
    ]


def text_evidence(question: str, strategy: str = "bm25") -> list[dict]:
    if strategy == "bm25":
        rows = bm25().search(question, k=8)
    elif strategy == "medcpt":
        rows = medcpt().search(question, k=8)
    elif strategy == "hybrid":
        candidates = reciprocal_rank_fusion(
            bm25().search(question, k=50), medcpt().search(question, k=50), k=60
        )[:30]
        rows = reranker().rerank(question, candidates, k=8)
    else:
        raise ValueError(f"Unknown text retrieval strategy: {strategy}")
    return rows_to_text_evidence(rows)


def graph_evidence(question: str) -> tuple[list[dict], list[dict]]:
    seeds = graph().link(question)
    paths = [path for path in graph().paths(seeds, question=question) if path["score"] >= 0.8]
    evidence = []
    for path in paths:
        verbalized = " · ".join(
            f"{next(node['name'] for node in path['nodes'] if node['id'] == edge['source_id'])}"
            f" —{edge['relation']}→ "
            f"{next(node['name'] for node in path['nodes'] if node['id'] == edge['target_id'])}"
            for edge in path["edges"]
        )
        evidence.append({
            "id": path["id"], "type": "graph", "title": "PrimeKG path",
            "snippet": verbalized, "score": path["score"], "nodes": path["nodes"], "edges": path["edges"],
            "provenance": {"kg": "PrimeKG", "revision": "Dataverse files 6180616/6180617"},
        })
    return evidence, seeds


@dataclass(frozen=True)
class Pipeline:
    id: str
    name: str
    description: str

    def run(self, question: str) -> dict:
        started = time.perf_counter()
        if self.id == "B0":
            evidence, budget = evidence_budget([], [])
            linked = []
        if self.id in {"B1", "B2", "B3"}:
            strategy = {"B1": "bm25", "B2": "medcpt", "B3": "hybrid"}[self.id]
            evidence, budget = evidence_budget(text_evidence(question, strategy), [])
            linked = []
        if self.id in {"G1", "G2"}:
            graphs, seeds = graph_evidence(question)
            texts = text_evidence(question, "hybrid") if self.id == "G2" else []
            evidence, budget = evidence_budget(texts, graphs)
            linked = seeds
        if self.id in {"B0", "B1", "B2", "B3", "G1", "G2"}:
            generation = generator().generate(question, evidence, closed_book=self.id == "B0")
            prompt_name = "answer_closed_book_v1.txt" if self.id == "B0" else "answer_v1.txt"
            integrity = validate_citations(generation.answer, evidence)
            registry = {item["id"]: item for item in evidence}
            cited = [registry[item_id] for item_id in integrity["valid_ids"]]
            answer = generation.answer
            if integrity["invented_ids"]:
                answer = "The generated answer failed citation-integrity validation; no medical answer is shown."
                cited = []
            return {
                "answer": answer, "citations": cited, "evidence": evidence,
                "details": {"pipeline": self.id, "linked_entities": linked,
                            "graph_paths": [item["snippet"] for item in evidence if item["type"] == "graph"],
                            "degraded": False, "degraded_reason": None, "budget": budget,
                            "citation_integrity": integrity,
                            "generator": {"provider": generation.provider, "model": generation.model,
                                          "cached": generation.cached},
                            "latency_ms": round((time.perf_counter() - started) * 1000)},
                "provenance": {"pipeline_id": self.id,
                    "pipeline_config_hash": file_hash(project_root() / "configs/pipelines.yaml"),
                    "prompt_hash": file_hash(project_root() / "configs/prompts" / prompt_name),
                    "data_manifest_hash": file_hash(project_root() / "data/manifests/files.json")},
            }
        raise ValueError(f"Unsupported pipeline: {self.id}")


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
    config_hash = file_hash(project_root() / "configs/pipelines.yaml")
    return [asdict(pipeline) | {"config_hash": config_hash} for pipeline in PIPELINES.values()]
