from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import atomic_json, sha256
from medrag_lab.settings import settings

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
MODEL_LICENSE = "public-domain"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MedCPTPaths:
    index: Path
    documents: Path
    metadata: Path


def paths(output_dir: Path | None = None) -> MedCPTPaths:
    root = output_dir or settings().medrag_index_dir / "medcpt"
    return MedCPTPaths(root / "index.faiss", root / "documents.json", root / "metadata.json")


def resolved_revision(model_id: str) -> str:
    from huggingface_hub import model_info

    revision = model_info(model_id).sha
    if not revision:
        raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
    return revision


def encode_articles(
    rows: list[dict[str, Any]], model: Any, tokenizer: Any, device: str, batch_size: int
) -> tuple[np.ndarray, int]:
    import torch

    batches: list[np.ndarray] = []
    model.eval()
    current_batch_size = batch_size
    start = 0
    while start < len(rows):
        batch = rows[start : start + current_batch_size]
        try:
            inputs = tokenizer(
                [(str(row.get("title", "")), str(row.get("text", ""))) for row in batch],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"),
            ):
                vectors = model(**inputs).last_hidden_state[:, 0, :]
                vectors = torch.nn.functional.normalize(vectors.float(), dim=1)
            batches.append(vectors.cpu().numpy().astype("float32"))
            start += len(batch)
            if start % 1_000 < len(batch) or start == len(rows):
                LOGGER.info(
                    "Encoded %s/%s articles (batch=%s)",
                    start,
                    len(rows),
                    current_batch_size,
                )
        except torch.OutOfMemoryError:
            if device != "cuda" or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
    return np.concatenate(batches), current_batch_size


def build_index(
    corpus: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int = 16,
    force: bool = False,
) -> MedCPTPaths:
    import faiss
    import torch
    from transformers import AutoModel, AutoTokenizer

    source = corpus or settings().medrag_data_dir / "corpus.jsonl"
    destination = paths(output_dir)
    if all(path.is_file() for path in vars(destination).values()) and not force:
        return destination
    article_revision = resolved_revision(ARTICLE_MODEL)
    query_revision = resolved_revision(QUERY_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL, revision=article_revision)
    model = AutoModel.from_pretrained(ARTICLE_MODEL, revision=article_revision).to(device)
    rows = list(iter_jsonl(source))
    vectors, effective_batch_size = encode_articles(rows, model, tokenizer, device, batch_size)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    destination.index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination.index))
    documents = [
        {
            "pmid": str(row["id"]),
            "title": str(row.get("title", "")),
            "text": str(row.get("text", "")),
            "url": str(row.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{row['id']}/"),
        }
        for row in rows
    ]
    atomic_json(destination.documents, documents)
    atomic_json(
        destination.metadata,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "corpus_sha256": sha256(source),
            "documents": len(documents),
            "dimensions": int(vectors.shape[1]),
            "similarity": "normalized_inner_product",
            "article_model": ARTICLE_MODEL,
            "article_revision": article_revision,
            "query_model": QUERY_MODEL,
            "query_revision": query_revision,
            "model_license": MODEL_LICENSE,
            "device": device,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
        },
    )
    return destination
