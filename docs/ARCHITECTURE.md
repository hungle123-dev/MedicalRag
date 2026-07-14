# MedicalRAG architecture

`MedicalRAGPipeline` is the shared gold-free runtime used by FastAPI and local product calls.
Experiment runners reuse the same retrieval/evidence/generation modules and can consume sealed
artifacts to keep controlled arms identical. Gold labels are opened only by evaluation code.

```text
Question
  -> original query
  -> BM25 + MedCPT reciprocal-rank fusion
  -> MedCPT document reranker (100 candidates)
  -> cross-encoder evidence selection (top 10 documents only)
  -> 600-token context, relevance order
  -> GPT-OSS-120B structured generation
  -> schema validation + PMID whitelist
  -> answer, citations, trace and latency
```

The top-10 evidence document cap is enforced in `medrag_lab/pipeline.py`; it matches the E04
evidence artifact consumed by the sealed E11 best-RAG evaluation. The product never receives
gold documents, gold snippets, ideal answers or question types.

```text
apps/api/                 FastAPI routes and runtime dependency loading
apps/web/                 React/Vite research UI
medrag_lab/data/          manifests, split firewall, EDA and loaders
medrag_lab/indexing/      BM25 and MedCPT indexes
medrag_lab/query/         original, MeSH and HyDE strategies
medrag_lab/retrieval/     dense, hybrid and reranker modules
medrag_lab/evidence/      snippet construction, ranking and context packing
medrag_lab/generation/    gateway, prompts, parser and schemas
medrag_lab/evaluation/    internal BioASQ-compatible metrics, statistics and judge diagnostics
medrag_lab/experiments/   controlled runners, gates and final freeze
medrag_lab/tracking/      MLflow helpers and trace storage
configs/                  historical experiment registry, judge and pipeline configs
reports/                  tracked machine-readable and reader-facing results
```

Scientific scope: positive-only, gold-conditioned BioASQ closed corpus. The held-out result
supports improved lexical overlap with ideal answers, not PubMed-wide retrieval, physician
validation, clinical correctness or clinical safety. E05/E08/E09 are development findings;
only the aggregate E11 pipeline contrasts are confirmatory.
