# Repository structure

The repository is a small monorepo. Directories are created only when their first working artifact exists.

```text
MedicalRag/
├── backend/                 # FastAPI app and shared research pipeline
│   ├── app/                 # routes, persistence, text/graph retrieval, generation
│   └── tests/               # small contract and end-to-end checks
├── frontend/                # React + Vite + TypeScript demo
│   └── src/                 # UI, API client, SSE handling
├── configs/                 # reviewed pipeline/experiment configs; no secrets
├── data/
│   └── manifests/           # pinned revisions, schemas, checksums, licenses
├── docs/
│   ├── adr/                 # architecture decisions
│   └── diagrams/            # renderable Mermaid sources
├── scripts/                 # thin data, experiment and evaluation entry points
├── artifacts/               # generated; demo/ and experiments/; gitignored
├── indexes/                 # generated BM25/vector indexes; gitignored
├── .env.example             # variable names and safe defaults only
└── README.md                # setup, commands, project status
```

## Ownership rules

- `backend/app` is the sole implementation of B0–G2. API and evaluation reuse it.
- `configs/` is reviewed input; generated runtime state belongs in `artifacts/`.
- `data/manifests/` is committed. Downloaded corpora, KG files, indexes, model weights, SQLite files and secrets are not.
- `frontend/` consumes the checked OpenAPI contract and never imports backend source.
- A frozen experiment artifact is append-only. Corrections create a new run ID.
- BioASQ end-to-end and PrimeKGQA component results use separate track directories and are never averaged into one score.

## Naming

- Pipeline IDs: `B0`, `B1`, `B2`, `B3`, `G1`, `G2`.
- Demo request IDs are UUIDs; research run IDs are descriptive slugs containing split, protocol version, population size and commit hash.
- Graph evidence IDs: `primekg:path:<sha256>` over canonical ordered path JSON.
- Text evidence IDs: stable corpus identifiers such as `PMID:<id>`.
- Track output: `artifacts/experiments/<track>/<run_id>/`, where `track` is `bioasq` or `primekgqa`.

## Minimal commands expected

The root README will expose one command each for backend, frontend, checks, a smoke experiment and evaluation. Do not add a task runner until repeated commands justify it.
