import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .pipelines import PIPELINES, list_pipelines
from .store import JobStore


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    pipeline_id: str = "G2"


def create_app(database: Path | None = None, artifacts: Path | None = None) -> FastAPI:
    data_dir = Path(os.getenv("MEDICAL_RAG_DATA_DIR", Path(__file__).parents[1] / "data"))
    store = JobStore(database or data_dir / "jobs.sqlite3", artifacts or data_dir / "artifacts")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        yield

    app = FastAPI(title="Medical Graph-RAG API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"status": "ready", "pipelines": len(PIPELINES)}

    @app.get("/api/v1/pipelines")
    def pipelines():
        return {"items": list_pipelines()}

    def execute(job_id: str, pipeline_id: str, question: str):
        job = store.get(job_id)
        if not job or job["status"] == "cancelled":
            return
        store.set_status(job_id, "running")
        try:
            result = PIPELINES[pipeline_id].run(question)
            if store.get(job_id)["status"] != "cancelled":
                store.set_status(job_id, "completed", result=result)
        except Exception as exc:  # boundary: persist failures for reproducible runs
            store.set_status(job_id, "failed", error=str(exc))

    @app.post("/api/v1/questions", status_code=202)
    def submit(payload: QuestionRequest, background_tasks: BackgroundTasks):
        if payload.pipeline_id not in PIPELINES:
            raise HTTPException(422, f"Unknown pipeline_id: {payload.pipeline_id}")
        job = store.create(payload.pipeline_id, payload.question.strip())
        background_tasks.add_task(execute, job["id"], payload.pipeline_id, payload.question.strip())
        return job

    @app.get("/api/v1/questions/{job_id}")
    def get_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Question job not found")
        return job

    @app.delete("/api/v1/questions/{job_id}")
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
            while True:
                job = store.get(job_id)
                if job["status"] != previous:
                    yield f"event: status\ndata: {json.dumps(job)}\n\n"
                    previous = job["status"]
                if job["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()

