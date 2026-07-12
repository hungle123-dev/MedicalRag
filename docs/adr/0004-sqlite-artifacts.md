# ADR 0004: SQLite metadata and JSON artifacts

- Status: accepted
- Date: 2026-07-12

## Context

The demo needs request state while research evaluation needs immutable raw outputs and provenance. A database server and message queue would add operations work without current scale requirements.

## Decision

Store request metadata and lifecycle in SQLite. Store full retrieval, generation, timing and provenance records as JSON artifacts, written by temporary file plus atomic rename. Use one Uvicorn worker with an in-process task registry and concurrency semaphore.

## Consequences

Completed runs are replayable without model calls and easy to inspect. Horizontal scaling and durable queued work are unsupported. Add a queue/database service only when multiple workers or machines become a measured requirement.

