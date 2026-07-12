# ADR 0002: FastAPI backend and React frontend

- Status: accepted
- Date: 2026-07-12

## Context

Research code and biomedical model tooling are Python-first. The demo needs accessible streaming UI states and typed API consumption, not a second implementation of the pipeline.

## Decision

Use a FastAPI monolith around the shared Python pipeline and React + Vite + TypeScript for the browser. Use local component state; add no global state library.

## Consequences

FastAPI publishes OpenAPI and runs the same B0–G2 code as evaluation. React owns presentation only. Two development processes are acceptable; the production demo may serve the compiled frontend from FastAPI.

