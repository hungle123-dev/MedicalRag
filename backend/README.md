# Backend

FastAPI runtime shared by the UI and research evaluator. B1/B2/B3 and G1/G2 use
the local real BM25, MedCPT/FAISS and PrimeKG indexes. The deterministic generator
is retained only for offline flow tests; copy `.env.example` to `.env` and set
the gateway key for credentialed inference.

```powershell
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
python -m pytest tests -q
```

Runtime data is stored in `backend/data/` unless `MEDICAL_RAG_DATA_DIR` is set.
