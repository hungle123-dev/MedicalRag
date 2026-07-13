import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, database: Path, artifacts: Path):
        database.parent.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.artifacts = artifacts
        self.lock = Lock()
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, pipeline_id TEXT NOT NULL, question TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT, error TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
            )
            connection.execute(
                "UPDATE jobs SET status='failed', error='SERVER_RESTARTED', updated_at=? "
                "WHERE status IN ('queued','running')", (utc_now(),)
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create(self, pipeline_id: str, question: str) -> dict:
        job_id, now = str(uuid4()), utc_now()
        with self.lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, 'queued', NULL, NULL, ?, ?)",
                (job_id, pipeline_id, question, now, now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def set_status(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None) -> bool:
        now = utc_now()
        if result:
            target = self.artifacts / f"{job_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, target)
        with self.lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(result) if result else None, error, now, job_id),
            )
        return cursor.rowcount == 1

    def cancel(self, job_id: str) -> bool:
        with self.lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? "
                "WHERE id = ? AND status IN ('queued', 'running')",
                (utc_now(), job_id),
            )
        return cursor.rowcount == 1
