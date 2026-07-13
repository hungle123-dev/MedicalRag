"""Build a resumable MedCPT article index for the frozen BioASQ corpus."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


SENTENCE = re.compile(r"(?<=[.!?])\s+")
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
ARTICLE_REVISION = "d05a736da4bb84ee4057b7f7999485be6ed85465"


def read_documents(corpus: Path, strategy: str, tokenizer) -> list[dict]:
    documents = []
    with corpus.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            title, text = row.get("title", ""), row.get("text", "")
            chunks = [text]
            if strategy == "C2":
                chunks, current = [], []
                sentences = SENTENCE.split(text)
                tokenized_sentences = tokenizer(sentences, add_special_tokens=False)["input_ids"]
                size = 0
                for sentence, token_ids in zip(sentences, tokenized_sentences):
                    if current and size + len(token_ids) > 256:
                        chunks.append(" ".join(current)); current = []
                        size = 0
                    current.append(sentence); size += len(token_ids)
                if current: chunks.append(" ".join(current))
            for chunk_index, chunk in enumerate(chunks):
                documents.append({
                    "id": str(row["id"]), "chunk_id": f"PMID:{row['id']}:C{chunk_index}",
                    "title": title, "text": chunk,
                    "url": row.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{row['id']}/",
                })
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/bioasq/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "indexes/medcpt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--strategy", choices=("C0", "C2"), default="C0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL, revision=ARTICLE_REVISION)
    documents = read_documents(args.corpus, args.strategy, tokenizer)
    count, dimension = len(documents), 768
    vectors_path = args.output / "article_vectors.f32"
    state_path = args.output / "build_state.json"
    metadata_path = args.output / "metadata.jsonl"
    if not metadata_path.exists():
        temporary = metadata_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in documents:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, metadata_path)

    mode = "r+" if vectors_path.exists() else "w+"
    vectors = np.memmap(vectors_path, dtype="float32", mode=mode, shape=(count, dimension))
    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("document_count") != count or state.get("strategy", "C0") != args.strategy:
            raise RuntimeError("Checkpoint corpus count differs; remove output and rebuild")
        completed = int(state.get("completed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ARTICLE_MODEL, revision=ARTICLE_REVISION).to(device).eval()
    started = time.perf_counter()
    for start in range(completed, count, args.batch_size):
        end = min(start + args.batch_size, count)
        pairs = [[row["title"], row["text"]] for row in documents[start:end]]
        encoded = tokenizer(
            pairs, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            batch = model(**encoded).last_hidden_state[:, 0, :].float().cpu().numpy()
        vectors[start:end] = batch
        if end % 256 < args.batch_size or end == count:
            vectors.flush()
            elapsed = time.perf_counter() - started
            atomic_json(state_path, {
                "status": "encoding", "document_count": count, "completed": end,
                "device": device, "batch_size": args.batch_size,
                "strategy": args.strategy,
                "elapsed_this_run_seconds": round(elapsed, 3),
                "documents_per_second": round((end - completed) / elapsed, 3),
            })
            print(f"encoded {end}/{count} ({(end-completed)/elapsed:.1f} docs/s)", flush=True)

    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(vectors))
    temporary_index = args.output / "articles.faiss.tmp"
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, args.output / "articles.faiss")
    atomic_json(state_path, {
        "status": "complete", "document_count": count, "completed": count,
        "dimension": dimension, "similarity": "inner_product", "device": device,
        "batch_size": args.batch_size, "article_encoder": ARTICLE_MODEL,
        "article_encoder_revision": ARTICLE_REVISION,
        "strategy": args.strategy,
    })
    print(f"completed {args.output / 'articles.faiss'}", flush=True)


if __name__ == "__main__":
    main()
