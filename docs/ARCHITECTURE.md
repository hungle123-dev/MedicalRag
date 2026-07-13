MedicalRag/
├── apps/
│ ├── api/
│ │ ├── main.py # FastAPI application
│ │ ├── dependencies.py # Shared pipeline/runtime loading
│ │ └── routes/
│ │ ├── answer.py
│ │ ├── compare.py
│ │ ├── experiments.py
│ │ └── traces.py
│ │
│ └── web/ # React + Vite
│ ├── src/
│ │ ├── components/
│ │ ├── pages/
│ │ ├── api.ts
│ │ └── App.tsx
│ ├── package.json
│ └── vite.config.ts
│
├── medrag_lab/
│ ├── data/
│ │ ├── audit.py # EDA, provenance, integrity
│ │ ├── loaders.py # JSONL loading
│ │ ├── manifests.py # Hashes and immutable manifest
│ │ ├── splits.py # Group-aware frozen populations
│ │ └── schemas.py # Corpus/question models
│ │
│ ├── indexing/
│ │ ├── bm25.py
│ │ └── medcpt.py
│ │
│ ├── query/
│ │ ├── original.py
│ │ ├── mesh.py
│ │ ├── hyde.py
│ │ └── iterative.py
│ │
│ ├── retrieval/
│ │ ├── sparse.py
│ │ ├── dense.py
│ │ ├── hybrid.py
│ │ └── reranker.py
│ │
│ ├── evidence/
│ │ ├── snippets.py
│ │ ├── chunking.py
│ │ ├── packing.py
│ │ └── citations.py
│ │
│ ├── generation/
│ │ ├── gateway.py # Direct OpenAI-compatible client
│ │ ├── prompts.py
│ │ ├── parser.py
│ │ └── schemas.py
│ │
│ ├── evaluation/
│ │ ├── retrieval.py
│ │ ├── bioasq.py
│ │ ├── semantic.py # BERTScore, not Ragas
│ │ ├── llm_panel.py
│ │ ├── statistics.py
│ │ └── errors.py
│ │
│ ├── experiments/
│ │ ├── registry.py
│ │ ├── runner.py
│ │ ├── gates.py
│ │ └── final.py
│ │
│ ├── tracking/
│ │ ├── mlflow_tracking.py
│ │ └── traces.py
│ │
│ ├── schemas.py # Shared API/pipeline contracts
│ ├── settings.py
│ └── pipeline.py # Shared end-to-end orchestrator
│
├── configs/
│ ├── experiments/
│ │ └── registry.yaml
│ ├── pipelines/
│ │ ├── bm25.yaml
│ │ ├── best.yaml
│ │ └── oracle.yaml
│ ├── judges/
│ │ └── panel.yaml
│ └── protocol.yaml
│
├── scripts/ # Thin entrypoints only
├── tests/
│ ├── data/
│ ├── evaluation/
│ ├── retrieval/
│ ├── generation/
│ └── api/
│
├── data/
│ ├── raw/ # Gitignored
│ ├── processed/ # Gitignored
│ └── manifests/ # Tracked
│
├── artifacts/ # Indexes, cache, raw responses; ignored
├── mlruns/ # MLflow runtime; ignored
├── results/ # Large generated results; ignored
├── reports/ # Tracked summaries/tables
├── docs/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
