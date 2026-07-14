from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes import answer, compare, experiments, traces
from medrag_lab.data.manifests import EXPECTED
from medrag_lab.settings import settings

app = FastAPI(
    title="MedicalRAG Research API",
    version="0.1.0",
    description="Closed-corpus BioASQ research system; not medical advice.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(answer.router)
app.include_router(compare.router)
app.include_router(experiments.router)
app.include_router(traces.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, object]:
    config = settings()
    data_ok = all((config.medrag_data_dir / filename).is_file() for filename in EXPECTED)
    gateway_ok = bool(config.openai_api_key and config.openai_base_url)
    dense_ok = (config.medrag_index_dir / "medcpt" / "index.faiss").is_file()
    return {
        "ready": data_ok and gateway_ok and dense_ok,
        "default_pipeline": "best_rag",
        "checks": {
            "pinned_data_present": data_ok,
            "gateway_configured": gateway_ok,
            "best_rag_medcpt_index_present": dense_ok,
        },
    }
