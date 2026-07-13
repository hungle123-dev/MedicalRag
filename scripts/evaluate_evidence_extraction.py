"""Compare gold-free answer-context extractors on frozen B3 rankings."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from statistics import paired_bootstrap

TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
BUDGET = 1800


def normalize(value: str) -> str:
    return " ".join(TOKEN.findall(value.casefold()))


def lexical_sentences(question: str, text: str, limit: int = 225) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE.split(text) if sentence.strip()]
    if not sentences:
        return text
    tokenized = [TOKEN.findall(sentence.casefold()) for sentence in sentences]
    scores = BM25Okapi(tokenized).get_scores(TOKEN.findall(question.casefold()))
    selected, words = [], 0
    for index in sorted(range(len(sentences)), key=lambda item: (-scores[item], item)):
        candidate = sentences[index].split()
        if words >= limit:
            break
        selected.append(index); words += min(len(candidate), limit - words)
    return " ".join(sentences[index] for index in sorted(selected))


def context(question: str, documents: list[dict], strategy: str) -> tuple[str, int]:
    parts, used = [], 0
    for row in documents[:8]:
        if used >= BUDGET:
            break
        if strategy == "P0_prefix_600_chars":
            extracted = row["text"][:600]
        elif strategy == "P1_full_abstract_budgeted":
            extracted = row["text"]
        elif strategy == "P2_lexical_sentence_windows":
            extracted = lexical_sentences(question, row["text"])
        else:
            raise ValueError(strategy)
        words = extracted.split()[:BUDGET - used]
        used += len(words)
        parts.append(f"{row.get('title', '')}\n{' '.join(words)}")
    return "\n".join(parts), used


def main() -> None:
    retrieval_path = ROOT / "artifacts/experiments/bioasq/bioasq_dev_b1_b2_b3_20260712/retrieval.json"
    rankings = {row["question_id"]: row["pipelines"]["B3"]["ranking"][:8]
                for row in json.loads(retrieval_path.read_text(encoding="utf-8"))["rows"]}
    with (ROOT / "data/raw/bioasq/corpus.jsonl").open(encoding="utf-8") as stream:
        corpus = {str(row["id"]): row for line in stream if (row := json.loads(line))}
    with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
        questions = {row["question_id"]: row for line in stream if (row := json.loads(line))
                     and row["question_id"] in rankings}

    strategies = ("P0_prefix_600_chars", "P1_full_abstract_budgeted",
                  "P2_lexical_sentence_windows")
    rows, elapsed = [], {strategy: 0.0 for strategy in strategies}
    for question_id in rankings:
        row = questions[question_id]
        documents = [corpus[pmid] for pmid in rankings[question_id]]
        retrieved_ids = set(rankings[question_id])
        snippets = [snippet for snippet in row["snippets"]
                    if snippet.get("beginSection") in {"abstract", "title"}
                    and snippet.get("document", "").rstrip("/").rsplit("/", 1)[-1] in retrieved_ids]
        for strategy in strategies:
            started = time.perf_counter()
            rendered, words = context(row["question"], documents, strategy)
            elapsed[strategy] += time.perf_counter() - started
            normalized_context = normalize(rendered)
            visible = [normalize(snippet["text"]) in normalized_context for snippet in snippets]
            rows.append({"question_id": question_id, "strategy": strategy,
                         "retrieved_gold_snippets": len(snippets),
                         "visible_gold_snippets": sum(visible),
                         "any_gold_snippet_visible": float(any(visible)),
                         "gold_snippet_visibility": sum(visible) / len(visible) if visible else 0.0,
                         "context_words": words})

    metrics = {}
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        total_snippets = sum(row["retrieved_gold_snippets"] for row in selected)
        metrics[strategy] = {
            "questions": len(selected),
            "questions_with_retrieved_gold": sum(row["retrieved_gold_snippets"] > 0 for row in selected),
            "question_any_visible_rate": round(sum(row["any_gold_snippet_visible"] for row in selected) /
                                               len(selected), 6),
            "retrieved_gold_snippet_visibility": round(
                sum(row["visible_gold_snippets"] for row in selected) / max(total_snippets, 1), 6),
            "mean_context_words": round(sum(row["context_words"] for row in selected) / len(selected), 2),
            "extraction_mean_ms": round(elapsed[strategy] * 1000 / len(selected), 3),
        }
    by_strategy = {strategy: {row["question_id"]: row for row in rows if row["strategy"] == strategy}
                   for strategy in strategies}
    paired = {}
    for strategy in strategies[1:]:
        ids = list(rankings)
        paired[f"{strategy}_minus_P0_any_visible"] = paired_bootstrap(
            [by_strategy[strategies[0]][qid]["any_gold_snippet_visible"] for qid in ids],
            [by_strategy[strategy][qid]["any_gold_snippet_visible"] for qid in ids])
    selected = max(strategies, key=lambda strategy: (
        metrics[strategy]["question_any_visible_rate"],
        metrics[strategy]["retrieved_gold_snippet_visibility"],
        -metrics[strategy]["extraction_mean_ms"],
    ))
    report = {
        "run_id": "bioasq_dev_evidence_extraction_20260713",
        "population": "frozen BioASQ dev-300 B3 top-8 rankings",
        "budget": "1,800 whitespace words; title visible; no gold used by extractors",
        "metrics": metrics, "paired_bootstrap": paired,
        "selection_rule": "highest any-gold visibility, then snippet visibility, then extraction latency",
        "selected": selected,
        "limitations": "Gold-snippet visibility is a component diagnostic, not final answer correctness.",
    }
    artifact = ROOT / "artifacts/experiments/bioasq" / report["run_id"]
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    target = ROOT / "data/manifests/bioasq_evidence_extraction_dev.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
