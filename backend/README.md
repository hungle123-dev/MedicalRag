# Backend

Minimal FastAPI contract for the research pipelines. The current B0–G2 runners
are deterministic placeholders so frontend and experiment integration can start
before model artifacts are downloaded.

```powershell
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
python -m unittest discover -s tests
```

Runtime data is stored in `backend/data/` unless `MEDICAL_RAG_DATA_DIR` is set.
