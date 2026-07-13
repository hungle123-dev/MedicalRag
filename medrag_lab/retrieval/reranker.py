from __future__ import annotations

import time
from typing import Any

from medrag_lab.schemas import RetrievedDocument

CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"


class MedCPTReranker:
    def __init__(self, device: str | None = None):
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        revision = model_info(CROSS_ENCODER).sha
        if not revision:
            raise RuntimeError("Could not resolve MedCPT cross-encoder revision")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER, revision=revision)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(
            CROSS_ENCODER, revision=revision
        ).to(self.device)
        self.model.eval()
        self.revision = revision
        self.last_effective_batch_size = 0

    def rerank(
        self, query: str, documents: list[RetrievedDocument], k: int = 20, batch_size: int = 128
    ) -> tuple[list[RetrievedDocument], float]:
        return self.rerank_many([(query, documents)], k=k, batch_size=batch_size)[0]

    def rerank_many(
        self,
        items: list[tuple[str, list[RetrievedDocument]]],
        k: int = 20,
        batch_size: int = 128,
    ) -> list[tuple[list[RetrievedDocument], float]]:
        """Batch pairs across questions while preserving an independent ranking per question."""
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not items:
            return []
        started = time.perf_counter()
        pairs = [
            (question_index, query, document)
            for question_index, (query, documents) in enumerate(items)
            for document in documents
        ]
        scored: list[list[tuple[float, RetrievedDocument]]] = [[] for _ in items]
        current_batch_size = batch_size
        start = 0
        while start < len(pairs):
            batch = pairs[start : start + current_batch_size]
            try:
                inputs = self.tokenizer(
                    [query for _, query, _ in batch],
                    [f"{document.title} {document.text}" for _, _, document in batch],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"
                    ),
                ):
                    logits = self.model(**inputs).logits.squeeze(-1).float().cpu().tolist()
                if not isinstance(logits, list):
                    logits = [float(logits)]
                for score, (question_index, _, document) in zip(
                    map(float, logits), batch, strict=True
                ):
                    scored[question_index].append((score, document))
                start += len(batch)
            except torch.OutOfMemoryError:
                if self.device != "cuda" or current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
        self.last_effective_batch_size = current_batch_size
        amortized_ms = (time.perf_counter() - started) * 1_000 / len(items)
        results = []
        for question_scores in scored:
            question_scores.sort(key=lambda item: -item[0])
            ranked = [
                document.model_copy(
                    update={"score": score, "rank": rank, "retriever": "medcpt_xenc"}
                )
                for rank, (score, document) in enumerate(question_scores[:k], 1)
            ]
            results.append((ranked, amortized_ms))
        return results
