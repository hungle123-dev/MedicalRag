# ADR 0004: SQLite metadata and JSON artifacts

- Status: accepted
- Date: 2026-07-12

## Context

The demo needs request state while research evaluation needs immutable raw outputs and provenance. A database server and message queue would add operations work without current scale requirements.

## Decision

Store request metadata, lifecycle and result JSON in SQLite. Store an inspectable job envelope plus full retrieval, generation, timing and provenance records as JSON artifacts, written by temporary file plus atomic rename. Use one Uvicorn worker with an in-process task registry and concurrency semaphore.

## Consequences

Completed generation runs are replayable without recalling the generator and easy to inspect. A non-cached machine judge remains an external model call. Horizontal scaling and durable queued work are unsupported. Add a queue/database service only when multiple workers or machines become a measured requirement.
