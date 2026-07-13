"""Run the pre-locked BioASQ BM25 chunking experiment on real dev data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bioasq"
MANIFESTS = ROOT / "data" / "manifests"
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN.findall(text)]


def sentence_chunks(text: str, limit: int = 256, tokenizer=None) -> list[str]:
    chunks, current, size = [], [], 0
    sentences = SENTENCE.split(text)
    lengths = ([len(ids) for ids in tokenizer(sentences, add_special_tokens=False)["input_ids"]]
               if tokenizer else [len(tokenize(sentence)) for sentence in sentences])
    for sentence, length in zip(sentences, lengths):
        if current and size + length > limit:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sentence.strip())
        size += length
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def build_documents(strategy: str) -> tuple[list[str], list[str]]:
    pmids, texts = [], []
    tokenizer = None
    if strategy == "C2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    for row in load_jsonl(RAW / "corpus.jsonl"):
        title = row.get("title", "")
        chunks = [row["text"]] if strategy == "C0" else sentence_chunks(row["text"], tokenizer=tokenizer)
        for chunk in chunks:
            pmids.append(str(row["id"]))
            texts.append(f"{title}. {chunk}")
    return pmids, texts


def collapse_top(scores: np.ndarray, pmids: list[str], k: int) -> list[str]:
    fetch = min(max(k * 5, 50), len(scores))
    indexes = np.argpartition(scores, -fetch)[-fetch:]
    indexes = indexes[np.argsort(scores[indexes])[::-1]]
    results, seen = [], set()
    for index in indexes:
        pmid = pmids[int(index)]
        if pmid not in seen:
            results.append(pmid)
            seen.add(pmid)
        if len(results) == k:
            break
    return results


def question_metrics(
    ranking: list[str], gold: set[str], k: int = 10, snippet_pmids: list[str] | None = None
) -> dict[str, float]:
    hits = [1 if pmid in gold else 0 for pmid in ranking[:k]]
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(gold), k)))
    return {
        "recall_at_5": sum(hits[:5]) / len(gold) if gold else 0.0,
        "recall_at_10": sum(hits) / len(gold) if gold else 0.0,
        "precision_at_5": sum(hits[:5]) / 5,
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "snippet_document_coverage_at_10": (
            sum(pmid in set(ranking[:k]) for pmid in snippet_pmids) / len(snippet_pmids)
            if snippet_pmids else 0.0
        ),
    }


def evaluate(strategy: str, questions: list[dict]) -> dict:
    started = time.perf_counter()
    pmids, texts = build_documents(strategy)
    tokenized = [tokenize(text) for text in texts]
    index = BM25Okapi(tokenized)
    rows = []
    for question in questions:
        scores = index.get_scores(tokenize(question["question"]))
        ranking = collapse_top(scores, pmids, 10)
        snippet_pmids = [str(item.get("document", "")).rstrip("/").split("/")[-1] for item in question.get("snippets", [])]
        metrics = question_metrics(ranking, set(map(str, question["relevant_passage_ids"])), snippet_pmids=snippet_pmids)
        rows.append({"question_id": question["question_id"], "ranking": ranking, "metrics": metrics})
    aggregate = {
        key: round(sum(row["metrics"][key] for row in rows) / len(rows), 6)
        for key in rows[0]["metrics"]
    }
    return {
        "strategy": strategy,
        "documents": len(texts),
        "questions": len(rows),
        "metrics": aggregate,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    dev = load_jsonl(RAW / "dev.jsonl")
    selected = random.Random(args.seed).sample(dev, min(args.questions, len(dev)))
    ids = [row["question_id"] for row in selected]
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    ids_path = MANIFESTS / "bioasq_dev_question_ids.json"
    ids_path.write_text(json.dumps({"seed": args.seed, "ids": ids}, indent=2), encoding="utf-8")
    results = [evaluate(strategy, selected) for strategy in ("C0", "C2")]
    run_id = "bioasq_dev_bm25_" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
    artifact = ROOT / "artifacts" / "experiments" / "bioasq" / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id, "seed": args.seed, "question_ids_file": str(ids_path.relative_to(ROOT)),
        "results": [{key: value for key, value in result.items() if key != "rows"} for result in results],
    }
    (MANIFESTS / "bioasq_bm25_dev.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
