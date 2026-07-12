# MedicalRag

Evidence-grounded medical question answering with controlled comparisons between strong Text-RAG and hybrid Text + PrimeKG retrieval.

## Scope

- Input: an English medical question.
- Output: a text answer with structured PubMed/PrimeKG evidence.
- Research core: BioASQ long-form QA; MedQA is supportive answer-only evaluation.
- Product demo: React frontend + FastAPI backend over the same frozen pipeline registry.
- Out of scope: crawling, clinical deployment, EHR uploads, LLM fine-tuning, microservices, Neo4j and GNNs.

## Architecture

```text
frontend/ React + TypeScript
       │ HTTP + SSE
backend/ FastAPI
       ├── B0–B3 text pipelines
       ├── G1/G2 PrimeKG pipelines
       ├── evidence/citation contracts
       └── SQLite metadata + JSON artifacts

configs/ frozen protocol and pipeline definitions
docs/    research plan, architecture, API and ADRs
```

See [the execution plan](docs/KE_HOACH_NGHIEN_CUU_MedicalGraphRAG.html) and [architecture](docs/ARCHITECTURE.md).

## Development

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The default scaffold uses deterministic mock pipelines so the full API/UI contract can be tested before downloading data or models.

## Checks

```powershell
cd backend
python -m unittest discover -s tests
cd ..\frontend
npm run build
```

## Repository rules

- Never commit datasets, indexes, model weights, `.env`, API keys or generated artifacts.
- Locked experiments are identified by data/config/prompt/code hashes.
- The retriever never receives MedQA answer options.
- Demo artifacts and experiment artifacts use separate namespaces.
- This is a research prototype, not medical advice.

## Status

Architecture scaffold. Real BioASQ/MedCPT/PrimeKG adapters are implemented only after the protocol values marked `MUST_FREEZE_W1` are verified and frozen.
