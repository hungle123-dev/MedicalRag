# Architecture diagrams

## Pipeline

```mermaid
flowchart TD
  Q[Medical question] --> R{Pipeline registry}
  R -->|B1/B3/G2| T[Text retrieval]
  R -->|G1/G2| L[Entity linking]
  T --> BM[BM25]
  T --> MC[MedCPT]
  BM --> F[RRF + reranking]
  MC --> F
  L --> P[PrimeKG paths: BioASQ 1–2 hop]
  F --> E[Evidence fusion and fixed budget]
  P --> E
  R -->|B0| G[Fixed generator]
  E --> G
  G --> A[Answer + citation map + provenance]
```

## Evaluation tracks

```mermaid
flowchart LR
  BQ[BioASQ questions] --> B3[B3 Text-RAG]
  BQ --> G2[G2 Text + PrimeKG]
  B3 --> BE[End-to-end paired evaluator]
  G2 --> BE
  PQ[PrimeKGQA question + answer + SPARQL] --> GR[Graph linker / retriever up to 3 hops]
  KG[(Matching PrimeKG RDF)] --> GR
  GR --> PE[Answer-set and execution evaluator]
```

## Request states

```mermaid
stateDiagram-v2
  [*] --> queued
  queued --> running
  queued --> cancelled
  running --> completed
  running --> failed
  running --> cancelled
  queued --> failed: server restart / dependency failure
  completed --> [*]
  failed --> [*]
  cancelled --> [*]
```
