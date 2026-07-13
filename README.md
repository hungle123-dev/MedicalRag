# Medical Text + Knowledge Graph RAG

Research prototype for English medical QA. Input is one text question; output is a grounded text answer, PubMed evidence, optional PrimeKG paths, provenance and an explicit insufficient-evidence response. It is not medical advice.

## What is implemented

- B0 closed-book diagnostic; B1 BM25; B2 MedCPT dense; B3 BM25 + MedCPT RRF + MedCPT Cross-Encoder.
- G1 PrimeKG-only diagnostic; E5 compares B3, G2, equal-budget extra-text X1 and a structurally valid unlinked-PrimeKG-path X2. Every arm matches B3's actual per-question whitespace-word budget.
- BioASQ end-to-end track and PrimeKGQA graph-component track. PrimeKGQA is QA evaluation data; PrimeKG is the one queried graph.
- FastAPI/SQLite/JSON-artifact backend and React/TypeScript UI with SSE, cancellation, evidence panels and B3/G2 comparison.
- Deterministic offline generator for tests; cached OpenAI-compatible gateway generator and blinded two-pass independent judge for real inference.

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

CPU-only install for flow verification:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run the app

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm ci
npm run dev
```

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and switch `MEDICAL_RAG_GENERATOR=gateway` for real generation. Keep `mock` for quota-free flow tests. Keys are loaded only by the backend, never accepted by the frontend, logged or committed.

The demo is intentionally local/loopback. It has no public-deployment authentication or per-user ownership. Do not expose it to the Internet and do not enter patient-identifiable data.

The frozen 20-question dev calibration selected `deepseek-v3.2` as generator and `cerebras/gpt-oss-120b` as the independent structured judge. DeepSeek scored higher on correctness/completeness; Qwen had fewer unsupported claims. The next E5 population excludes all 20 selection questions.

## Experiments and checks

```powershell
python scripts/evaluate_bm25.py --questions 300
python scripts/analyze_retrieval.py artifacts/experiments/bioasq/bioasq_dev_bm25_243e7ce0f400/retrieval.json
python scripts/evaluate_medcpt.py
python scripts/primekgqa_gate.py --count 100
python scripts/evaluate_primekgqa.py --split val --sample 300
python scripts/evaluate_evidence_extraction.py
python scripts/audit_bioasq_graph_coverage.py
python scripts/run_bioasq_end_to_end.py --split dev --questions 80
python scripts/evaluate_answer_run.py artifacts/experiments/bioasq/<completed-v9-run> --workers 4
python scripts/export_human_review.py artifacts/experiments/bioasq/<locked-run> --reviewer a
python scripts/export_human_review.py artifacts/experiments/bioasq/<locked-run> --reviewer b
python scripts/analyze_human_review.py artifacts/human_review/reviewer_a.csv artifacts/human_review/reviewer_b.csv artifacts/human_review/reviewer_a_mapping.json artifacts/human_review/reviewer_b_mapping.json --output artifacts/human_review/analysis.json --run artifacts/experiments/bioasq/<locked-run>

cd backend
python -m pytest tests -q
cd ../frontend
npm run build
```

The pinned PrimeKGQA SPARQL compatibility gate currently fails (3% non-empty execution on the 100-query smoke) because published RDF node IRIs do not directly map to the pinned Dataverse CSV node indices. Therefore the project reports normalized-pattern fallback metrics and does **not** claim valid SPARQL execution accuracy.

Start with the Vietnamese [step-by-step overview](docs/TONG_QUAN_TUNG_BUOC_MedicalGraphRAG.html), then the [actual results](docs/RESULTS.md), [detailed research plan](docs/KE_HOACH_NGHIEN_CUU_MedicalGraphRAG.html), [architecture](docs/ARCHITECTURE.md), [bias/leakage controls](docs/BIAS_AND_LEAKAGE_CONTROLS.md), [human review protocol](docs/HUMAN_EVALUATION.md), and [error taxonomy](docs/ERROR_TAXONOMY.md).

## Current status and external blocker

- Gateway calibration is complete: generator `deepseek-v3.2`, independent exploratory judge `cerebras/gpt-oss-120b`. E5-v2 uses B3/G2/X1/X2 and a corrected full-abstract text baseline.
- The completed v8 dev-80 run is retained as a design pilot: 0 generation failures and equal total budgets, but a strict post-budget audit found malformed X2 paths in 10/43 graph-positive questions. Atomic-path v9 is required and no v8 answer-quality claim is permitted.
- Atomic-path v9 completed 80 questions / 320 outputs with 0 generation failures, 43 graph-positive questions and 196/196 valid X2 slots. Exploratory blinded judging did not support graph benefit: G2−B3 correctness was −0.05 (95% paired bootstrap CI −0.15 to 0.0375), and G2 did not beat X1 or X2. No tuning follows this result.
- The locked BioASQ eval-340 execution is run only after E5-v2 dev validation and a final clean freeze commit.
- Two qualified medical reviewers must complete the frozen 100-question blinded review; AI-generated labels cannot replace them.
- MedQA, crawling, EHR/PHI, clinical deployment, model fine-tuning, Neo4j, GNNs and billing fallback are out of core scope.
