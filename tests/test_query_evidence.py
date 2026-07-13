import json
from pathlib import Path
from types import SimpleNamespace

from medrag_lab.evidence.chunking import fixed_token_chunks
from medrag_lab.evidence.packing import source_diverse, strongest_in_middle
from medrag_lab.evidence.snippets import Snippet, document_snippet_candidates, sentence_windows
from medrag_lab.experiments import evidence
from medrag_lab.experiments.generation import _frozen_snippets
from medrag_lab.query.mesh import MeshExpander
from medrag_lab.schemas import RetrievedDocument


def test_mesh_expansion_is_conservative_and_gold_free(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"1","title":"x","text":"x","mesh_terms":["Breast Neoplasms"]}\n',
        encoding="utf-8",
    )
    expander = MeshExpander(corpus)
    assert expander.expand("How are breast neoplasms treated?")[0].endswith("Breast Neoplasms")
    assert expander.expand("Unrelated question") == ["Unrelated question"]


def test_chunk_order_and_diversity() -> None:
    document = RetrievedDocument(
        pmid="1",
        title="Title",
        text=" ".join(["biomedical"] * 100),
        url="https://example.test/1",
        score=1,
        rank=1,
        retriever="test",
    )
    assert len(fixed_token_chunks([document], size=32, overlap=8)) > 1
    snippets = [
        Snippet(str(index // 2), "t", str(index), 1 - index / 10, "u") for index in range(4)
    ]
    assert strongest_in_middle(snippets)[1] == snippets[0]
    assert [item.pmid for item in source_diverse(snippets)] == ["0", "1"]


def test_evidence_chunks_preserve_bioasq_character_offsets() -> None:
    text = "First sentence.  Second sentence! Third sentence?"
    document = RetrievedDocument(
        pmid="1",
        title="Title",
        text=text,
        url="u",
        score=1,
        rank=1,
        retriever="test",
    )
    window = sentence_windows(document, size=2)[0]
    assert window.begin == 0
    assert window.end is not None
    assert text[window.begin : window.end] == window.text
    for chunk in fixed_token_chunks([document], size=32, overlap=8):
        assert chunk.begin is not None and chunk.end is not None
        assert text[chunk.begin : chunk.end] == chunk.text


def test_e04_candidate_pool_is_title_first() -> None:
    documents = [
        RetrievedDocument(
            pmid=str(index),
            title=f"Title {index}",
            text="First sentence. Second sentence.",
            url="u",
            score=1,
            rank=index,
            retriever="test",
        )
        for index in (1, 2)
    ]
    candidates = document_snippet_candidates(documents)
    assert [(item.pmid, item.section) for item in candidates[:2]] == [
        ("1", "title"),
        ("2", "title"),
    ]


def test_merge_evidence_shards_requires_exact_coverage(tmp_path, monkeypatch) -> None:
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    (tmp_path / "data" / "manifests" / "splits.json").write_text(
        json.dumps({"selection4849": ["q1", "q2"], "freeze_hash": "f"})
    )
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(
        evidence, "settings", lambda: SimpleNamespace(medrag_artifact_dir=tmp_path / "artifacts")
    )
    paths = []
    for question_id in ("q1", "q2"):
        path = tmp_path / f"{question_id}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "snippets": [],
                    "metrics": {"precision": 1, "recall": 1, "f1": 1, "gold_pmid_recall": 1},
                    "latency_ms": 1,
                    "failed": False,
                }
            )
            + "\n"
        )
        paths.append(path)
    result = evidence.merge_evidence_shards(paths, "selection4849", "sentence3_cross_encoder")
    assert result["metrics"]["questions"] == 2
    assert Path(result["artifacts"]["predictions"]).name == "predictions.jsonl"


def test_frozen_e04_annotations_preserve_order_and_offsets() -> None:
    annotations = [
        {
            "document": "https://pubmed.ncbi.nlm.nih.gov/123/",
            "text": "evidence",
            "beginSection": "abstract",
            "offsetInBeginSection": 4,
            "offsetInEndSection": 12,
        }
    ]
    snippets = _frozen_snippets(annotations, {"123": {"title": "Title", "url": "u"}})
    assert [(item.pmid, item.text, item.begin, item.end) for item in snippets] == [
        ("123", "evidence", 4, 12)
    ]
