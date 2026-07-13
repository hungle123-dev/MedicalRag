from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


MEDCPT_REVISIONS = {
    "ncbi/MedCPT-Query-Encoder": "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc",
    "ncbi/MedCPT-Cross-Encoder": "71caf65d4927987813984f54c284405a13fcca49",
}


def reciprocal_rank_fusion(*rankings: list[dict], k: int = 60) -> list[dict]:
    """Fuse ranked lists by document id without calibrating incompatible scores."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            doc_id = str(row["id"])
            if doc_id not in fused:
                fused[doc_id] = dict(row) | {"rrf_score": 0.0, "retrievers": []}
            fused[doc_id]["rrf_score"] += 1.0 / (k + rank)
            source = row.get("retriever")
            if source and source not in fused[doc_id]["retrievers"]:
                fused[doc_id]["retrievers"].append(source)
    return sorted(fused.values(), key=lambda row: (-row["rrf_score"], str(row["id"])))


class MedCPTIndex:
    """FAISS inner-product index paired with the official MedCPT query encoder."""

    def __init__(self, directory: Path, device: str | None = None):
        self.directory = directory
        self.index = faiss.read_index(str(directory / "articles.faiss"))
        with (directory / "metadata.jsonl").open(encoding="utf-8") as stream:
            self.documents = [json.loads(line) for line in stream]
        if self.index.ntotal != len(self.documents):
            raise ValueError("MedCPT FAISS index and metadata length differ")
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load_encoder(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        selected = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "ncbi/MedCPT-Query-Encoder"
        revision = MEDCPT_REVISIONS[model_id]
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(model_id, revision=revision).to(selected).eval()
        self.device = selected

    def search(self, question: str, k: int = 8) -> list[dict]:
        import torch

        self._load_encoder()
        encoded = self._tokenizer(
            [question], truncation=True, padding=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            vector = self._model(**encoded).last_hidden_state[:, 0, :].float().cpu().numpy()
        scores, indexes = self.index.search(np.ascontiguousarray(vector), min(k, self.index.ntotal))
        return [
            self.documents[int(index)]
            | {"score": float(score), "retriever": "medcpt"}
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]


class MedCPTReranker:
    """Official MedCPT cross-encoder reranker, loaded lazily for runtime use."""

    def __init__(self, device: str | None = None, batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "ncbi/MedCPT-Cross-Encoder"
        revision = MEDCPT_REVISIONS[model_id]
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision).to(self.device).eval()

    def rerank(self, question: str, candidates: list[dict], k: int = 8) -> list[dict]:
        import torch

        self._load()
        scored = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            pairs = [[question, f"{row['title']} {row['text']}"] for row in batch]
            encoded = self._tokenizer(
                pairs, truncation=True, padding=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits.reshape(-1).float().cpu().tolist()
            scored.extend(row | {"rerank_score": float(score)} for row, score in zip(batch, logits))
        return sorted(scored, key=lambda row: (-row["rerank_score"], str(row["id"])))[:k]
