from pathlib import Path

import pytest

from medrag_lab.generation.schemas import GatewayResult, GeneratedAnswer
from medrag_lab.indexing.bm25 import BM25Index
from medrag_lab.pipeline import EVIDENCE_DOCUMENT_LIMIT, MedicalRAGPipeline, load_pipeline_config
from medrag_lab.retrieval.reranker import CROSS_ENCODER_REVISION
from medrag_lab.schemas import AnswerRequest, RetrievedDocument
from medrag_lab.tracking.traces import TraceStore


class FakeGateway:
    def generate(self, **_: object) -> GatewayResult:
        return GatewayResult(
            answer=GeneratedAnswer(
                predicted_type="factoid",
                exact_answer=["BRCA1"],
                ideal_answer="BRCA1 is associated with hereditary breast cancer risk.",
                citation_pmids=["1", "999"],
                evidence_support_score=0.9,
            ),
            model="test-model",
            provider="test",
            input_tokens=20,
            output_tokens=10,
            latency_ms=1,
        )


class FailingGateway:
    def generate(self, **_: object) -> GatewayResult:
        raise TimeoutError("synthetic unit-test timeout")


def test_product_defaults_to_frozen_heldout_winner() -> None:
    assert AnswerRequest(question="What is BRCA1?").pipeline_id == "best_rag"
    assert load_pipeline_config("best_rag")["retriever"] == "rrf_rerank"


def test_reranker_batch_size_is_frozen_to_measured_value() -> None:
    assert load_pipeline_config("rrf_rerank_rag")["rerank_batch_size"] == 64
    assert CROSS_ENCODER_REVISION == "71caf65d4927987813984f54c284405a13fcca49"


def test_cross_encoder_evidence_uses_same_top_ten_document_cap_as_e11(monkeypatch) -> None:
    pipeline = object.__new__(MedicalRAGPipeline)
    pipeline.config = {"evidence_strategy": "sentence3_cross_encoder", "snippet_limit": 20}
    pipeline.reranker = object()
    seen: list[str] = []

    def fake_rank(_question, candidates, _reranker, limit):
        seen.extend(item.pmid for item in candidates)
        return candidates[:limit], 0.0

    monkeypatch.setattr("medrag_lab.pipeline.rank_snippets_cross_encoder", fake_rank)
    documents = [
        RetrievedDocument(
            pmid=str(index),
            title="t",
            text="sentence one. sentence two. sentence three.",
            url=f"https://example.test/{index}",
            score=1.0,
            rank=index + 1,
            retriever="test",
        )
        for index in range(25)
    ]
    pipeline._evidence("question", documents)
    assert set(seen) == {str(index) for index in range(EVIDENCE_DOCUMENT_LIMIT)}


def test_shared_pipeline_filters_hallucinated_citations(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"1","title":"BRCA1","text":"BRCA1 mutations increase hereditary breast cancer risk.","url":"https://example.test/1"}\n',
        encoding="utf-8",
    )
    pipeline = MedicalRAGPipeline(
        "bm25_rag",
        gateway=FakeGateway(),  # type: ignore[arg-type]
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
        index=BM25Index.build(corpus),
    )
    result = pipeline.answer(
        AnswerRequest(question="What does BRCA1 affect?", pipeline_id="bm25_rag")
    )
    assert [citation.pmid for citation in result.citations] == ["1"]
    assert pipeline.trace_store.get(result.trace_id) is not None


def test_shared_pipeline_traces_provider_failure(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"1","title":"BRCA1","text":"BRCA1 evidence.","url":"u"}\n',
        encoding="utf-8",
    )
    store = TraceStore(tmp_path / "failures.sqlite3")
    pipeline = MedicalRAGPipeline(
        "bm25_rag",
        gateway=FailingGateway(),  # type: ignore[arg-type]
        trace_store=store,
        index=BM25Index.build(corpus),
    )
    with pytest.raises(TimeoutError):
        pipeline.answer(AnswerRequest(question="What is BRCA1?", pipeline_id="bm25_rag"))
    with store._connect() as connection:  # failure has no response object carrying its trace ID
        payload = connection.execute("SELECT payload FROM traces").fetchone()[0]
    assert '"failed": true' in payload
