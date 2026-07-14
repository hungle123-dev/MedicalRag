from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import yaml

from medrag_lab.evidence.chunking import fixed_token_chunks
from medrag_lab.evidence.packing import (
    pack_context,
    serialize_context,
    source_diverse,
    strongest_in_middle,
)
from medrag_lab.evidence.snippets import (
    Snippet,
    document_snippet_candidates,
    rank_snippets,
    rank_snippets_cross_encoder,
)
from medrag_lab.generation.gateway import GatewayClient
from medrag_lab.generation.prompts import SYSTEM_PROMPT, answer_prompt, prompt_hash
from medrag_lab.indexing.bm25 import BM25Index
from medrag_lab.query.mesh import MeshExpander
from medrag_lab.schemas import AnswerRequest, AnswerResponse, Citation, RetrievedDocument
from medrag_lab.settings import ROOT, settings
from medrag_lab.tracking.traces import TraceStore


@dataclass(frozen=True)
class PreparedContext:
    query: str
    documents: list[RetrievedDocument]
    snippets: list[Snippet]
    packed: list[Snippet]
    serialized_context: str
    retrieval_ms: float


class MedicalRAGPipeline:
    """Single product/experiment orchestrator; gold labels never enter this class."""

    def __init__(
        self,
        pipeline_id: str = "best_rag",
        *,
        gateway: GatewayClient | None = None,
        trace_store: TraceStore | None = None,
        index: BM25Index | None = None,
        config_override: dict[str, Any] | None = None,
    ):
        self.pipeline_id = pipeline_id
        self.config = load_pipeline_config(pipeline_id) | (config_override or {})
        self.gateway = gateway or GatewayClient()
        self.trace_store = trace_store or TraceStore()
        self.index = index or _load_or_build_bm25(str(self.config["bm25_recipe"]))
        self.dense = None
        self.reranker = None
        retriever = self.config.get("retriever", "bm25")
        if retriever in {"medcpt", "rrf", "rrf_rerank"}:
            from medrag_lab.retrieval.dense import MedCPTRetriever

            self.dense = MedCPTRetriever()
        if (
            retriever == "rrf_rerank"
            or self.config.get("evidence_strategy") == "sentence3_cross_encoder"
        ):
            from medrag_lab.retrieval.reranker import MedCPTReranker

            self.reranker = MedCPTReranker()
        self.mesh_expander = (
            MeshExpander(settings().medrag_data_dir / "corpus.jsonl")
            if self.config.get("query_strategy") == "mesh"
            else None
        )

    def prepare_context(
        self,
        question: str,
        *,
        documents_override: list[RetrievedDocument] | None = None,
        evidence_override: list[Snippet] | None = None,
    ) -> PreparedContext:
        """Gold-free retrieval and evidence preparation reusable across controlled arms."""
        query = question
        if self.mesh_expander:
            query = self.mesh_expander.expand(query)[0]
        retrieval_k = min(int(self.config["retrieval_k"]), len(self.index.documents))
        retriever = self.config.get("retriever", "bm25")
        if documents_override is not None:
            documents, retrieval_ms = documents_override, 0.0
        elif evidence_override is not None:
            documents, retrieval_ms = [], 0.0
        elif retriever == "bm25":
            documents, retrieval_ms = self.index.search(query, retrieval_k)
        elif retriever == "medcpt" and self.dense:
            documents, retrieval_ms = self.dense.retrieve(query, retrieval_k)
        elif self.dense:
            from medrag_lab.retrieval.hybrid import reciprocal_rank_fusion

            sparse, sparse_ms = self.index.search(query, retrieval_k)
            dense, dense_ms = self.dense.retrieve(query, retrieval_k)
            documents = reciprocal_rank_fusion(sparse, dense)[:retrieval_k]
            retrieval_ms = sparse_ms + dense_ms
            if self.reranker:
                documents, rerank_ms = self.reranker.rerank(
                    query,
                    documents,
                    retrieval_k,
                    batch_size=int(self.config.get("rerank_batch_size", 64)),
                )
                retrieval_ms += rerank_ms
        else:
            raise ValueError(f"Unsupported retriever: {retriever}")
        snippets = (
            list(evidence_override)
            if evidence_override is not None
            else self._evidence(question, documents)
        )
        if self.config.get("diversity") == "one_per_pmid":
            snippets = source_diverse(snippets)
        context, packed = pack_context(snippets, int(self.config["context_token_budget"]))
        if self.config.get("context_order") == "strongest_middle":
            packed = strongest_in_middle(packed)
            context = serialize_context(packed)
        return PreparedContext(query, documents, snippets, packed, context, retrieval_ms)

    def answer(
        self,
        request: AnswerRequest,
        *,
        documents_override: list[RetrievedDocument] | None = None,
        evidence_override: list[Snippet] | None = None,
        system_prompt_override: str | None = None,
    ) -> AnswerResponse:
        if request.pipeline_id != self.pipeline_id:
            raise ValueError("Request pipeline_id does not match initialized pipeline")
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        prepared = self.prepare_context(
            request.question,
            documents_override=documents_override,
            evidence_override=evidence_override,
        )
        user_prompt = answer_prompt(request.question, prepared.serialized_context)
        system_prompt = system_prompt_override or SYSTEM_PROMPT
        try:
            generated = self.gateway.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.get("generator_model") or None,
                max_output_tokens=int(self.config["max_output_tokens"]),
            )
        except Exception as exc:
            self.trace_store.put(
                trace_id,
                self.pipeline_id,
                {
                    "request": request.model_dump(),
                    "query": prepared.query,
                    "retrieval_ms": prepared.retrieval_ms,
                    "packed_evidence": [vars(item) for item in prepared.packed],
                    "serialized_context": prepared.serialized_context,
                    "prompt_hash": prompt_hash(system_prompt, user_prompt),
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                },
            )
            raise
        evidence_by_pmid = {snippet.pmid: snippet for snippet in prepared.packed}
        citations = [
            Citation(
                pmid=pmid,
                title=evidence_by_pmid[pmid].title,
                snippet=evidence_by_pmid[pmid].text,
                url=evidence_by_pmid[pmid].url,
            )
            for pmid in generated.answer.citation_pmids
            if pmid in evidence_by_pmid
        ]
        latency_ms = (time.perf_counter() - started) * 1_000
        response = AnswerResponse(
            predicted_type=generated.answer.predicted_type,
            exact_answer=generated.answer.exact_answer,
            ideal_answer=generated.answer.ideal_answer,
            citations=citations,
            abstained=generated.answer.abstained,
            evidence_support_score=generated.answer.evidence_support_score,
            trace_id=trace_id,
            pipeline_id=self.pipeline_id,
            latency_ms=latency_ms,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            model=generated.model,
            attempts=generated.attempts,
            estimated_cost_usd=generated.estimated_cost_usd,
        )
        self.trace_store.put(
            trace_id,
            self.pipeline_id,
            {
                "request": request.model_dump(),
                "query": prepared.query,
                "retrieval_ms": prepared.retrieval_ms,
                "retrieved": [item.model_dump() for item in prepared.documents],
                "packed_evidence": [vars(item) for item in prepared.packed],
                "serialized_context": prepared.serialized_context,
                "prompt_hash": prompt_hash(system_prompt, user_prompt),
                "experimental_override": {
                    "documents": documents_override is not None,
                    "evidence": evidence_override is not None,
                    "system_prompt": system_prompt_override is not None,
                },
                "cached": generated.cached,
                "raw_response": generated.raw_response,
                "response": response.model_dump(),
            },
        )
        return response

    def _evidence(self, question: str, documents: list[RetrievedDocument]) -> list[Snippet]:
        strategy = self.config.get("evidence_strategy", "sentence3")
        limit = int(self.config["snippet_limit"])
        if strategy == "sentence3":
            return rank_snippets(question, documents, limit)
        if strategy == "sentence3_cross_encoder" and self.reranker:
            return rank_snippets_cross_encoder(
                question, document_snippet_candidates(documents), self.reranker, limit
            )[0]
        if strategy == "fixed256":
            chunks = fixed_token_chunks(documents)
            terms = set(question.casefold().split())
            chunks.sort(key=lambda item: -len(terms & set(item.text.casefold().split())))
            return chunks[:limit]
        if strategy == "full_abstract":
            return [
                Snippet(item.pmid, item.title, item.text, item.score, item.url)
                for item in documents[:limit]
            ]
        raise ValueError(f"Unsupported evidence strategy: {strategy}")


def load_pipeline_config(pipeline_id: str) -> dict[str, Any]:
    path = ROOT / "configs" / "pipelines" / f"{pipeline_id}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown pipeline: {pipeline_id}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("pipeline_id") != pipeline_id:
        raise ValueError(f"Pipeline ID mismatch in {path}")
    return value


def _load_or_build_bm25(recipe: str) -> BM25Index:
    path = settings().medrag_index_dir / f"bm25-{recipe}.pkl"
    if path.is_file():
        return BM25Index.load(path)
    index = BM25Index.build(settings().medrag_data_dir / "corpus.jsonl", recipe)  # type: ignore[arg-type]
    index.save(path)
    return index
