"""These tests read the REAL downloaded MIRAGE + MedRAG-textbooks data.
Skipped automatically if the data hasn't been downloaded (CI / fresh clone).
"""
from pathlib import Path

import pytest

from medgraphrag.data.mirage_loader import load_mirage, SUBTASKS
from medgraphrag.data.corpus_loader import load_textbook_corpus

MIRAGE_PATH = "data/raw/mirage_benchmark.json"
TEXTBOOKS_DIR = "data/raw/medrag_textbooks/chunk"

pytestmark = pytest.mark.skipif(
    not Path(MIRAGE_PATH).exists(),
    reason="real MIRAGE data not downloaded (run scripts/download_data.py)",
)


def test_loads_all_5_subtasks_with_correct_total():
    qs = load_mirage(MIRAGE_PATH)
    assert len(qs) == 7663
    seen_prefixes = {q.qid.split("_")[0] for q in qs}
    assert seen_prefixes == set(SUBTASKS)


def test_per_subtask_counts_match_known_sizes():
    expected = {"medqa": 1273, "medmcqa": 4183, "pubmedqa": 500, "bioasq": 618, "mmlu": 1089}
    for st, n in expected.items():
        qs = load_mirage(MIRAGE_PATH, subtask=st)
        assert len(qs) == n, f"{st}: expected {n}, got {len(qs)}"


def test_every_question_has_answer_in_options():
    qs = load_mirage(MIRAGE_PATH, subtask="medqa")
    for q in qs[:50]:
        assert q.answer in q.options


def test_textbook_corpus_loads_and_has_content():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, limit=200)
    assert len(corpus) == 200
    sample = next(iter(corpus.values()))
    assert len(sample) > 20


def test_textbook_corpus_can_filter_by_book():
    corpus = load_textbook_corpus(TEXTBOOKS_DIR, books=["Pathology_Robbins"], limit=50)
    assert len(corpus) == 50
    assert all(k.startswith("Pathology_Robbins_") for k in corpus)
