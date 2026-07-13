from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from medrag_lab.indexing.medcpt import QUERY_MODEL, MedCPTPaths, build_index, paths
from medrag_lab.schemas import RetrievedDocument


class MedCPTRetriever:
    def __init__(self, index_paths: MedCPTPaths | None = None, device: str | None = None):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.paths = index_paths or paths()
        if not all(path.is_file() for path in vars(self.paths).values()):
            self.paths = build_index()
        metadata = json.loads(self.paths.metadata.read_text(encoding="utf-8"))
        self.documents: list[dict[str, Any]] = json.loads(
            self.paths.documents.read_text(encoding="utf-8")
        )
        self.index = faiss.read_index(str(self.paths.index))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        revision = metadata["query_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL, revision=revision)
        self.model = AutoModel.from_pretrained(QUERY_MODEL, revision=revision).to(self.device)
        self.model.eval()

    def _encode(self, queries: list[str]) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            queries, truncation=True, padding=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"),
        ):
            vectors = self.model(**inputs).last_hidden_state[:, 0, :]
            vectors = torch.nn.functional.normalize(vectors.float(), dim=1)
        return vectors.cpu().numpy().astype("float32")

    def retrieve(self, query: str, k: int = 100) -> tuple[list[RetrievedDocument], float]:
        if not 1 <= k <= len(self.documents):
            raise ValueError("k must be within corpus size")
        started = time.perf_counter()
        scores, indexes = self.index.search(self._encode([query]), k)
        documents = [
            RetrievedDocument(
                **self.documents[int(index)],
                score=float(scores[0, rank - 1]),
                rank=rank,
                retriever="medcpt",
            )
            for rank, index in enumerate(indexes[0], 1)
        ]
        return documents, (time.perf_counter() - started) * 1_000
