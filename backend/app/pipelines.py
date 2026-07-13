import os
import time
import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

import yaml

from .retrieval import BM25Index
from .graph import PrimeKGIndex
from .medcpt import MedCPTIndex, MedCPTReranker, reciprocal_rank_fusion
from .generator import create_generator, validate_citations
from .controls import matched_extra_text, matched_random_paths


_bm25: BM25Index | None = None
_graph: PrimeKGIndex | None = None
_medcpt: MedCPTIndex | None = None
_reranker: MedCPTReranker | None = None
_generator = None
_init_lock = Lock()


def project_root() -> Path:
    return Path(os.getenv("MEDICAL_RAG_ROOT", Path(__file__).parents[2]))


@lru_cache(maxsize=1)
def pipeline_defaults() -> dict:
    """Load the frozen runtime values instead of silently duplicating config constants."""
    payload = yaml.safe_load((project_root() / "configs/pipelines.yaml").read_text(encoding="utf-8"))
    return payload["defaults"]


def bm25_index_path() -> Path:
    return Path(os.getenv("MEDICAL_RAG_BM25_INDEX", project_root() / "indexes" / "bm25_c0.pkl"))


def graph_index_path() -> Path:
    return Path(os.getenv("MEDICAL_RAG_GRAPH_INDEX", project_root() / "indexes" / "primekg.sqlite3"))


def medcpt_index_path() -> Path:
    return Path(os.getenv("MEDICAL_RAG_MEDCPT_INDEX", project_root() / "indexes" / "medcpt"))


def bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        with _init_lock:
            if _bm25 is None:
                path = bm25_index_path()
                if not path.exists():
                    raise RuntimeError(f"BM25 index unavailable: run python scripts/build_indexes.py ({path})")
                _bm25 = BM25Index.load(path)
    return _bm25


def graph() -> PrimeKGIndex:
    global _graph
    if _graph is None:
        with _init_lock:
            if _graph is None:
                path = graph_index_path()
                if not path.exists():
                    raise RuntimeError(f"PrimeKG index unavailable: run python scripts/build_graph_index.py ({path})")
                _graph = PrimeKGIndex(path)
    return _graph


def medcpt() -> MedCPTIndex:
    global _medcpt
    if _medcpt is None:
        with _init_lock:
            if _medcpt is None:
                path = medcpt_index_path()
                if not (path / "articles.faiss").exists():
                    raise RuntimeError(f"MedCPT index unavailable: run python scripts/build_medcpt_index.py ({path})")
                _medcpt = MedCPTIndex(path)
    return _medcpt


def reranker() -> MedCPTReranker:
    global _reranker
    if _reranker is None:
        with _init_lock:
            if _reranker is None:
                _reranker = MedCPTReranker()
    return _reranker


def generator():
    global _generator
    if _generator is None:
        with _init_lock:
            if _generator is None:
                _generator = create_generator(project_root())
    return _generator


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def trim_item(item: dict, word_limit: int) -> tuple[dict, int]:
    words = item.get("snippet", "").split()
    used = min(len(words), max(word_limit, 0))
    return item | {"snippet": " ".join(words[:used])}, used


def evidence_budget(texts: list[dict], graphs: list[dict], *, word_budget: int = 1800,
                    max_items: int = 8, graph_word_budget: int = 540) -> tuple[list[dict], dict]:
    """Apply an explicit word-proxy budget; hybrid arms may use extra evidence slots."""
    # ponytail: provider tokenizers are unavailable offline; replace proxy when the generator API exposes countTokens.
    selected_graphs: list[dict] = []
    selected_texts: list[dict] = []
    graph_used = text_used = 0
    graph_limit = min(graph_word_budget, word_budget)
    for item in graphs[:5]:
        if len(selected_graphs) == max_items or graph_used >= graph_limit:
            break
        trimmed, used = trim_item(item, min(graph_limit - graph_used, 180))
        if used:
            selected_graphs.append(trimmed); graph_used += used
    for item in texts:
        if len(selected_graphs) + len(selected_texts) == max_items or graph_used + text_used >= word_budget:
            break
        trimmed, used = trim_item(item, word_budget - graph_used - text_used)
        if used:
            selected_texts.append(trimmed); text_used += used
    selected = []
    for index in range(max(len(selected_texts), len(selected_graphs))):
        if index < len(selected_texts): selected.append(selected_texts[index])
        if index < len(selected_graphs): selected.append(selected_graphs[index])
    return selected, {
        "word_count_method": "whitespace_split_v1", "word_budget": word_budget,
        "max_items": max_items,
        "ordering": "rank_interleave_text_first_v1",
        "graph_words_requested": graph_limit if graphs else 0, "graph_words_actual": graph_used,
        "text_words_actual": text_used, "evidence_items": len(selected),
    }


def rows_to_text_evidence(rows: list[dict]) -> list[dict]:
    """Expose the retrieved C0 abstract; the global budget performs the only truncation."""
    return [
        {
            "id": f"PMID:{row['id']}", "type": "text", "title": row["title"],
            "snippet": row["text"], "url": row["url"],
            "source": "PubMed", "pmid": str(row["id"]),
            "score": row.get("rerank_score", row.get("rrf_score", row["score"])),
            "retrievers": row.get("retrievers", [row.get("retriever")]),
        }
        for row in rows
    ]


def text_evidence(question: str, strategy: str = "bm25", k: int = 8) -> list[dict]:
    defaults = pipeline_defaults()
    if strategy == "bm25":
        rows = bm25().search(question, k=k)
    elif strategy == "medcpt":
        rows = medcpt().search(question, k=k)
    elif strategy == "hybrid":
        candidates = reciprocal_rank_fusion(
            bm25().search(question, k=defaults["bm25_fetch_k"]),
            medcpt().search(question, k=defaults["medcpt_fetch_k"]),
            k=defaults["rrf_k"],
        )[:defaults["rerank_pool_k"]]
        rows = reranker().rerank(question, candidates, k=k)
    else:
        raise ValueError(f"Unknown text retrieval strategy: {strategy}")
    return rows_to_text_evidence(rows)


def path_to_evidence(path: dict) -> dict:
    verbalized = " · ".join(
        f"{next(node['name'] for node in path['nodes'] if node['id'] == edge['source_id'])}"
        f" —{edge['relation']}→ "
        f"{next(node['name'] for node in path['nodes'] if node['id'] == edge['target_id'])}"
        for edge in path["edges"]
    )
    return {
        "id": path["id"], "type": "graph", "title": "PrimeKG path",
        "snippet": verbalized, "score": path["score"], "hop_count": path["hop_count"],
        "nodes": path["nodes"], "edges": path["edges"],
        "provenance": {"kg": "PrimeKG", "revision": "Dataverse files 6180616/6180617"},
    }


def graph_evidence(question: str, *, limit: int = 5, threshold: float | None = 0.8) -> tuple[list[dict], list[dict]]:
    defaults = pipeline_defaults()
    seeds = graph().link(question, limit=defaults["max_seed_entities"])
    paths = graph().paths(seeds, question=question, limit=limit,
                          max_hops=defaults["bioasq_graph_hops"])
    if threshold is not None:
        paths = [path for path in paths if path["score"] >= threshold]
    return [path_to_evidence(path) for path in paths], seeds


def evidence_words(items: list[dict]) -> int:
    return sum(len(item.get("snippet", "").split()) for item in items)


def background_control_evidence(target_graphs: list[dict], seeds: list[dict], seed: int) -> list[dict]:
    excluded_nodes = {str(node["id"]) for item in target_graphs for node in item.get("nodes", [])}
    excluded_nodes.update(str(item["id"]) for item in seeds)
    return [path_to_evidence(path) for path in graph().background_paths(
        [item["hop_count"] for item in target_graphs], seed=seed,
        exclude_node_ids=excluded_nodes,
    )]


def build_e5_arms(question: str, seed: int) -> dict[str, dict]:
    """Build B3/G2 and two matched controls once, before any answer is generated."""
    defaults = pipeline_defaults()
    text_k = defaults["hybrid_max_items"]
    base_k = defaults["text_final_k"]
    max_words = defaults["evidence_total_words"]
    graph_words_limit = defaults["graph_max_words"]
    texts = text_evidence(question, "hybrid", k=text_k)
    base_texts = texts[:base_k]
    target_budget = min(max_words, evidence_words(base_texts))
    b3, b3_budget = evidence_budget(base_texts, [], word_budget=target_budget, max_items=base_k)
    relevant, seeds = graph_evidence(question, limit=defaults["final_paths"],
                                     threshold=defaults["graph_path_quality_threshold"])
    g2, g2_budget = evidence_budget(base_texts, relevant, word_budget=target_budget,
                                    max_items=text_k, graph_word_budget=min(graph_words_limit, target_budget))
    selected_graphs = [item for item in g2 if item["type"] == "graph"]
    graph_words = evidence_words(selected_graphs)
    if not selected_graphs:
        return {arm: {"evidence": b3, "budget": b3_budget, "linked": seeds,
                      "graph_retrieval_positive": False, "control_complete": True}
                for arm in ("B3", "G2", "X1", "X2")}
    extra = matched_extra_text(texts[base_k:text_k], selected_graphs)
    x1, x1_budget = evidence_budget(base_texts, extra, word_budget=target_budget,
                                    max_items=text_k, graph_word_budget=graph_words)
    candidates = background_control_evidence(selected_graphs, seeds, seed)
    random_paths = matched_random_paths(candidates, selected_graphs, seed)
    x2, x2_budget = evidence_budget(base_texts, random_paths, word_budget=target_budget,
                                    max_items=text_k, graph_word_budget=graph_words)
    return {
        "B3": {"evidence": b3, "budget": b3_budget, "linked": [], "graph_retrieval_positive": True,
               "control_complete": True},
        "G2": {"evidence": g2, "budget": g2_budget, "linked": seeds, "graph_retrieval_positive": True,
               "control_complete": True},
        "X1": {"evidence": x1, "budget": x1_budget, "linked": seeds, "graph_retrieval_positive": True,
               "control_complete": len(extra) == len(selected_graphs)},
        "X2": {"evidence": x2, "budget": x2_budget, "linked": seeds, "graph_retrieval_positive": True,
               "control_complete": len(random_paths) == len(selected_graphs)},
    }


def generate_from_evidence(question: str, pipeline_id: str, arm: dict, started: float | None = None) -> dict:
    started = started or time.perf_counter()
    evidence, budget, linked = arm["evidence"], arm["budget"], arm["linked"]
    generation = generator().generate(question, evidence, closed_book=pipeline_id == "B0")
    prompt_name = "answer_closed_book_v1.txt" if pipeline_id == "B0" else "answer_v1.txt"
    integrity = validate_citations(generation.answer, evidence)
    registry = {item["id"]: item for item in evidence}
    cited = [registry[item_id] for item_id in integrity["valid_ids"]]
    answer = generation.answer
    if integrity["invented_ids"]:
        answer = "The generated answer failed citation-integrity validation; no medical answer is shown."
        cited = []
    return {
        "answer": answer, "citations": cited, "evidence": evidence,
        "details": {"pipeline": pipeline_id, "linked_entities": linked,
                    "graph_paths": [item["snippet"] for item in evidence if item["type"] == "graph"],
                    "degraded": False, "degraded_reason": None, "budget": budget,
                    "graph_retrieval_positive": arm.get("graph_retrieval_positive"),
                    "control_complete": arm.get("control_complete", True),
                    "citation_integrity": integrity,
                    "generator": {"provider": generation.provider, "model": generation.model,
                                  "cached": generation.cached, "response_model": generation.response_model,
                                  "system_fingerprint": generation.system_fingerprint,
                                  "usage": generation.usage},
                    "latency_ms": round((time.perf_counter() - started) * 1000)},
        "provenance": {"pipeline_id": pipeline_id,
            "pipeline_config_hash": file_hash(project_root() / "configs/pipelines.yaml"),
            "prompt_hash": file_hash(project_root() / "configs/prompts" / prompt_name),
            "data_manifest_hash": file_hash(project_root() / "data/manifests/files.json")},
    }


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
        elif self.id in {"B1", "B2"}:
            strategy = {"B1": "bm25", "B2": "medcpt"}[self.id]
            evidence, budget = evidence_budget(text_evidence(question, strategy), [])
            linked = []
        elif self.id == "B3":
            texts = text_evidence(question, "hybrid")
            evidence, budget = evidence_budget(
                texts, [], word_budget=min(pipeline_defaults()["evidence_total_words"], evidence_words(texts)))
            linked = []
        elif self.id == "G1":
            graphs, seeds = graph_evidence(question)
            evidence, budget = evidence_budget([], graphs)
            linked = seeds
        elif self.id == "G2":
            texts = text_evidence(question, "hybrid")
            graphs, linked = graph_evidence(question)
            target = min(pipeline_defaults()["evidence_total_words"], evidence_words(texts))
            evidence, budget = evidence_budget(texts, graphs, word_budget=target,
                                                max_items=pipeline_defaults()["hybrid_max_items"],
                                                graph_word_budget=min(pipeline_defaults()["graph_max_words"], target))
        else:
            raise ValueError(f"Unsupported pipeline: {self.id}")
        arm = {"evidence": evidence, "budget": budget, "linked": linked,
               "graph_retrieval_positive": any(item["type"] == "graph" for item in evidence),
               "control_complete": True}
        return generate_from_evidence(question, self.id, arm, started)


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
