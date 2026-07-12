# ADR 0001: Use a small monorepo

- Status: accepted
- Date: 2026-07-12

## Context

The same team changes research pipelines, API contracts and the demo during a five-week project. Separate repositories add version coordination without independent deployment needs.

## Decision

Keep backend, frontend, configs, manifests and documentation in one repository. Generated data, indexes, weights and artifacts remain outside Git.

## Consequences

One commit can update a pipeline and its UI/API contract. CI and ownership stay simple. Repository size is controlled by manifests and `.gitignore`, not Git LFS by default.

