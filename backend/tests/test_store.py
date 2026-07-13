import tempfile
from pathlib import Path

from app.store import JobStore


def test_artifact_write_is_complete_and_has_no_temporary_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = JobStore(root / "jobs.db", root / "artifacts")
        job = store.create("B0", "A valid question?")
        store.set_status(job["id"], "completed", result={"answer": "done"})
        assert (root / "artifacts" / f"{job['id']}.json").read_text(encoding="utf-8")
        assert not list((root / "artifacts").glob("*.tmp"))


def test_restart_fails_incomplete_jobs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); store = JobStore(root / "jobs.db", root / "artifacts")
        job = store.create("B0", "A valid question?")
        restarted = JobStore(root / "jobs.db", root / "artifacts").get(job["id"])
        assert restarted["status"] == "failed" and restarted["error"] == "SERVER_RESTARTED"


def test_cancelled_job_cannot_be_overwritten_by_worker():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = JobStore(root / "jobs.db", root / "artifacts")
        job = store.create("B0", "What is asthma?")
        assert store.set_status(job["id"], "running", allowed_from={"queued"})
        assert store.cancel(job["id"])
        assert not store.set_status(job["id"], "failed", error="late worker",
                                    allowed_from={"running"})
        assert store.get(job["id"])["status"] == "cancelled"
