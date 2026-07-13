from pathlib import Path

from medrag_lab.evidence.chunking import fixed_token_chunks
from medrag_lab.evidence.packing import source_diverse, strongest_in_middle
from medrag_lab.evidence.snippets import Snippet, sentence_windows
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
