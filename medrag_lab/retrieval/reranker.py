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

    def rerank(
        self, query: str, documents: list[RetrievedDocument], k: int = 20, batch_size: int = 16
    ) -> tuple[list[RetrievedDocument], float]:
        import torch

        started = time.perf_counter()
        scored: list[tuple[float, RetrievedDocument]] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                [f"{item.title} {item.text}" for item in batch],
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
            scored.extend(zip(map(float, logits), batch, strict=True))
        scored.sort(key=lambda item: -item[0])
        result = [
            document.model_copy(update={"score": score, "rank": rank, "retriever": "medcpt_xenc"})
            for rank, (score, document) in enumerate(scored[:k], 1)
        ]
        return result, (time.perf_counter() - started) * 1_000
