from pathlib import Path

import pytest

from medrag_lab.generation.schemas import GatewayResult, GeneratedAnswer
from medrag_lab.indexing.bm25 import BM25Index
from medrag_lab.pipeline import MedicalRAGPipeline, load_pipeline_config
from medrag_lab.retrieval.reranker import CROSS_ENCODER_REVISION
from medrag_lab.schemas import AnswerRequest
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


def test_reranker_batch_size_is_frozen_to_measured_value() -> None:
    assert load_pipeline_config("rrf_rerank_rag")["rerank_batch_size"] == 64
    assert CROSS_ENCODER_REVISION == "71caf65d4927987813984f54c284405a13fcca49"


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
