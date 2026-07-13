from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medrag_lab.settings import ROOT


class TraceStore:
    def __init__(self, path: Path | None = None):
        self.path = path or ROOT / "artifacts" / "traces.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                pipeline_id TEXT NOT NULL, payload TEXT NOT NULL)"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def put(self, trace_id: str, pipeline_id: str, payload: dict[str, Any]) -> None:
        safe_payload = _redact(payload)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?)",
                (
                    trace_id,
                    datetime.now(UTC).isoformat(),
                    pipeline_id,
                    json.dumps(safe_payload, ensure_ascii=False),
                ),
            )

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT trace_id, created_at, pipeline_id, payload FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "trace_id": row["trace_id"],
            "created_at": row["created_at"],
            "pipeline_id": row["pipeline_id"],
            **json.loads(row["payload"]),
        }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(term in key.casefold() for term in ("key", "secret"))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
