import asyncio
import json
import logging
import os
from threading import BoundedSemaphore
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .pipelines import (PIPELINES, bm25, bm25_index_path, graph, graph_index_path,
                        list_pipelines, medcpt, medcpt_index_path)
from .store import JobStore
from .env import load_dotenv


ROOT = Path(__file__).parents[2]
load_dotenv(ROOT)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=3, max_length=2000)
    pipeline_id: Literal["B0", "B1", "B2", "B3", "G1", "G2"] = "G2"
    client_request_id: str | None = Field(default=None, max_length=128)
    run_kind: Literal["demo"] = "demo"


class JobResponse(BaseModel):
    id: str
    pipeline_id: str
    question: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class SubmissionResponse(JobResponse):
    stream_url: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    pipelines: dict[str, bool]
    dependencies: dict[str, object]


def create_app(database: Path | None = None, artifacts: Path | None = None) -> FastAPI:
    configured_data_dir = os.getenv("MEDICAL_RAG_DATA_DIR")
    data_dir = (Path(configured_data_dir).expanduser().resolve() if configured_data_dir
                else Path(__file__).parents[1] / "data")
    store = JobStore(database or data_dir / "jobs.sqlite3", artifacts or data_dir / "artifacts")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        yield

    app = FastAPI(title="Medical Graph-RAG API", version="0.1.0", lifespan=lifespan)
    origins = [value.strip() for value in os.getenv(
        "MEDICAL_RAG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if value.strip()]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"],
    )
    capacity = BoundedSemaphore(int(os.getenv("MEDICAL_RAG_MAX_CONCURRENCY", "2")))

    @app.get("/health", response_model=dict[str, str])
    @app.get("/api/v1/health", response_model=dict[str, str])
    def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResponse)
    @app.get("/api/v1/ready", response_model=ReadinessResponse)
    def ready():
        bm25_ready = graph_ready = dense_ready = False
        database_ready = store.ping()
        try:
            bm25_ready = bm25_index_path().is_file() and len(bm25().documents) > 0
        except Exception:  # readiness boundary: corrupted or incompatible local index
            pass
        try:
            if graph_index_path().is_file():
                with graph().connect() as connection:
                    required = {"nodes", "edges", "node_degrees"}
                    present = {row[0] for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")}
                    graph_ready = required.issubset(present)
        except Exception:
            pass
        dense_root = medcpt_index_path()
        try:
            dense_ready = ((dense_root / "articles.faiss").is_file()
                           and (dense_root / "metadata.jsonl").is_file()
                           and medcpt().index.ntotal == len(medcpt().documents))
        except Exception:
            pass
        selected_generator = os.getenv("MEDICAL_RAG_GENERATOR", "mock").casefold()
        generator_ready = (selected_generator == "mock" or
                           (selected_generator == "gateway" and bool(os.getenv("OPENAI_API_KEY"))
                            and bool(os.getenv("OPENAI_BASE_URL")) and bool(os.getenv("GATEWAY_GENERATOR_MODEL")))
                           or (selected_generator == "gemini" and bool(os.getenv("GEMINI_API_KEY"))))
        ready = bm25_ready and graph_ready and dense_ready and generator_ready and database_ready
        availability = {"B0": generator_ready, "B1": bm25_ready and generator_ready,
                        "B2": dense_ready and generator_ready,
                        "B3": bm25_ready and dense_ready and generator_ready,
                        "G1": graph_ready and generator_ready,
                        "G2": bm25_ready and dense_ready and graph_ready and generator_ready}
        return JSONResponse({"status": "ready" if ready else "degraded", "pipelines": availability,
                "dependencies": {"bm25": bm25_ready, "primekg": graph_ready, "medcpt": dense_ready,
                                 "generator_configured": generator_ready,
                                 "generator_provider": selected_generator,
                                 "provider_reachability_checked": False,
                                 "database": database_ready}},
                status_code=200 if ready else 503)

    @app.get("/api/v1/pipelines", response_model=dict[str, list[dict]])
    def pipelines():
        return {"items": list_pipelines()}

    def execute(job_id: str, pipeline_id: str, question: str):
        job = store.get(job_id)
        if not job or job["status"] == "cancelled":
            return
        if not capacity.acquire(timeout=1):
            store.set_status(job_id, "failed", error="PIPELINE_BUSY", allowed_from={"queued"})
            return
        if not store.set_status(job_id, "running", allowed_from={"queued"}):
            capacity.release()
            return
        try:
            result = PIPELINES[pipeline_id].run(question)
            store.set_status(job_id, "completed", result=result, allowed_from={"running"})
        except Exception as exc:  # boundary: persist failures for reproducible runs
            logger.exception("Pipeline execution failed for job_id=%s pipeline_id=%s", job_id, pipeline_id)
            store.set_status(job_id, "failed", error=f"PIPELINE_FAILED:{type(exc).__name__}",
                             allowed_from={"running"})
        finally:
            capacity.release()

    @app.post("/api/v1/questions", status_code=202, response_model=SubmissionResponse)
    def submit(payload: QuestionRequest, background_tasks: BackgroundTasks):
        if payload.pipeline_id not in PIPELINES:
            raise HTTPException(422, f"Unknown pipeline_id: {payload.pipeline_id}")
        job = store.create(payload.pipeline_id, payload.question.strip())
        background_tasks.add_task(execute, job["id"], payload.pipeline_id, payload.question.strip())
        return job | {"stream_url": f"/api/v1/questions/{job['id']}/events"}

    @app.get("/api/v1/questions/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Question job not found")
        return job

    @app.delete("/api/v1/questions/{job_id}", response_model=JobResponse)
    def cancel(job_id: str):
        if not store.get(job_id):
            raise HTTPException(404, "Question job not found")
        if not store.cancel(job_id):
            raise HTTPException(409, "Only queued or running jobs can be cancelled")
        return store.get(job_id)

    @app.get("/api/v1/questions/{job_id}/events")
    async def events(job_id: str):
        if not store.get(job_id):
            raise HTTPException(404, "Question job not found")

        async def stream():
            previous = None
            event_id = 0
            ticks = 0
            while True:
                job = store.get(job_id)
                if job["status"] != previous:
                    event_id += 1
                    yield f"id: {event_id}\nevent: status\ndata: {json.dumps(job)}\n\n"
                    previous = job["status"]
                if job["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.25)
                ticks += 1
                if ticks % 60 == 0:
                    yield ": heartbeat\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
