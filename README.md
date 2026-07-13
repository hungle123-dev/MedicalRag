# Medical Text + Knowledge Graph RAG

Research prototype for English medical QA. Input is one text question; output is a grounded text answer, PubMed evidence, optional PrimeKG paths, provenance and an explicit insufficient-evidence response. It is not medical advice.

## What is implemented

- B0 closed-book diagnostic; B1 BM25; B2 MedCPT dense; B3 BM25 + MedCPT RRF + MedCPT Cross-Encoder.
- G1 PrimeKG-only diagnostic; G2 B3 plus 1–2 hop PrimeKG evidence under the same 1,800-token/8-item budget.
- BioASQ end-to-end track and PrimeKGQA graph-component track. PrimeKGQA is QA evaluation data; PrimeKG is the one queried graph.
- FastAPI/SQLite/JSON-artifact backend and React/TypeScript UI with SSE, cancellation, evidence panels and B3/G2 comparison.
- Deterministic offline generator for tests; cached Gemini and blinded two-pass Groq judge adapters for real inference when credentials exist.

## Verified real data

| Data | Role | Actual count |
|---|---|---:|
| BioASQ text corpus | Retrieval knowledge base | 49,513 abstracts |
| BioASQ dev / locked eval | End-to-end questions and gold labels | 5,049 / 340 |
| PrimeKG | Queried knowledge graph | 129,375 nodes / 8,100,498 directed edges |
| PrimeKGQA train / val / test | Graph-component benchmark | 51,220 / 17,074 / 17,074 |

Exact URLs, revisions, hashes and licenses are in `data/manifests/`. Raw datasets, indexes, model weights, keys and run artifacts are intentionally gitignored.

## Reproduce locally

Python 3.12 and Node are used in the verified Windows environment.

```powershell
python -m pip install -r requirements-research.txt
python scripts/data_pipeline.py download all
python scripts/data_pipeline.py eda
python scripts/audit_data.py
python scripts/build_indexes.py
python scripts/build_graph_index.py
```

NVIDIA GPU setup used here (RTX 3050 Laptop, driver-compatible CUDA runtime):

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/build_medcpt_index.py --strategy C0 --output indexes/medcpt --batch-size 8
python scripts/build_medcpt_index.py --strategy C2 --output indexes/medcpt_c2 --batch-size 8
```

CPU MedCPT is supported but impractical for full indexing. PyTorch wheels are platform-specific and are not in `requirements-research.txt`.

## Run the app

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

The default `MEDICAL_RAG_GENERATOR=mock` spends no quota. For real generation set `MEDICAL_RAG_GENERATOR=gemini` and provide `GEMINI_API_KEY` in the environment. Judge evaluation separately requires `GROQ_API_KEY`. Keys are never accepted by the frontend or committed.

## Experiments and checks

```powershell
python scripts/evaluate_bm25.py --questions 300
python scripts/analyze_retrieval.py artifacts/experiments/bioasq/bioasq_dev_bm25_243e7ce0f400/retrieval.json
python scripts/evaluate_medcpt.py
python scripts/primekgqa_gate.py --count 100
python scripts/evaluate_primekgqa.py --split val --sample 300

cd backend
python -m pytest tests -q
cd ../frontend
npm run build
```

The pinned PrimeKGQA SPARQL compatibility gate currently fails (3% non-empty execution on the 100-query smoke) because published RDF node IRIs do not directly map to the pinned Dataverse CSV node indices. Therefore the project reports normalized-pattern fallback metrics and does **not** claim valid SPARQL execution accuracy.

Start with the Vietnamese [step-by-step overview](docs/TONG_QUAN_TUNG_BUOC_MedicalGraphRAG.html), then the [actual results](docs/RESULTS.md), [detailed research plan](docs/KE_HOACH_NGHIEN_CUU_MedicalGraphRAG.html), [architecture](docs/ARCHITECTURE.md), [bias/leakage controls](docs/BIAS_AND_LEAKAGE_CONTROLS.md), [human review protocol](docs/HUMAN_EVALUATION.md), and [error taxonomy](docs/ERROR_TAXONOMY.md).

## Current external blockers

- No Gemini/Groq credentials are present, so real generator/judge inference and the locked B3-vs-G2 answer comparison are not run.
- Two qualified medical reviewers must complete the frozen 100-question blinded review; AI draft labels cannot replace them.
- MedQA, crawling, EHR/PHI, clinical deployment, model fine-tuning, Neo4j, GNNs and billing fallback are out of core scope.
