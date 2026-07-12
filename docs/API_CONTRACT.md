# API contract v1

Base path: `/api/v1`. FastAPI's exported and hashed `openapi.json` is authoritative once implementation begins.

## Endpoints

| Method | Route | Result |
|---|---|---|
| GET | `/health` | liveness without loading models |
| GET | `/ready` | dependency and per-pipeline availability; 503 when unavailable |
| GET | `/pipelines` | allowlisted B0–G2 and config hashes |
| POST | `/questions` | validate and submit one demo question; returns 202 |
| GET | `/questions/{id}/events` | SSE request events |
| GET | `/questions/{id}` | persisted state/result |
| DELETE | `/questions/{id}` | cancel an active demo request; never delete artifacts |

## Submit

```json
{
  "question": "Why can propranolol worsen asthma?",
  "pipeline_id": "G2",
  "client_request_id": "018f5c9a-e03f-7b9d-a2f2-acde48001122",
  "run_kind": "demo"
}
```

`question` is 1–2000 characters after trimming. `pipeline_id` is an enum. `run_kind` must be `demo`; unknown fields are rejected.

```json
{
  "request_id": "018f5c9b-1904-7603-9987-acde48001122",
  "status": "queued",
  "stream_url": "/api/v1/questions/018f5c9b-1904-7603-9987-acde48001122/events"
}
```

## State and errors

States are `queued`, `running`, `completed`, `failed`, `cancelled`. All errors use a stable code and safe message:

```json
{"code":"PIPELINE_UNAVAILABLE","message":"G2 is currently unavailable.","retryable":true}
```

Initial codes: `INVALID_REQUEST`, `NOT_FOUND`, `PIPELINE_UNAVAILABLE`, `DEPENDENCY_TIMEOUT`, `GENERATOR_FAILED`, `SERVER_RESTARTED`, `CANCELLED`, `INTERNAL_ERROR`.

## Server-Sent Events

Responses use `text/event-stream`, monotonically increasing `id`, heartbeat comments and one terminal event. Reconnect may send `Last-Event-ID`.

```text
id: 4
event: evidence
data: {"request_id":"...","evidence":[...]}

id: 5
event: token
data: {"text":"Propranolol"}

id: 6
event: completed
data: {"request_id":"...","answer":"...","citation_map":{},"evidence":[],"metrics":{},"provenance":{}}
```

Event types are `status`, `evidence`, `token`, `completed`, `error`. Cancellation ends with a terminal status/event and closes the stream.

## Evidence

Text evidence:

```json
{
  "id":"PMID:12345678", "type":"text", "rank":1, "score":0.92,
  "title":"...", "snippet":"...", "source":"PubMed",
  "pmid":"12345678", "doi":"10.0000/example",
  "url":"https://pubmed.ncbi.nlm.nih.gov/12345678/"
}
```

Graph evidence:

```json
{
  "id":"primekg:path:<sha256>", "type":"graph", "rank":1, "score":0.81,
  "nodes":[{"id":"...","name":"propranolol","type":"drug"}],
  "edges":[{"source_id":"...","relation":"...","target_id":"...","source_dataset":"..."}],
  "verbalized":"...", "provenance":{"kg":"PrimeKG","revision":"..."}
}
```

The server validates URLs and evidence IDs. Every answer marker `[n]` maps through `citation_map` to an existing evidence ID. Missing or invented IDs are flagged and never turned into links.

## Completed result

```json
{
  "request_id":"...", "status":"completed", "answer":"...",
  "citation_map":{"1":"PMID:12345678"}, "evidence":[],
  "metrics":{"latency_ms":1200},
  "provenance":{"pipeline_id":"G2","config_hash":"...","model":"...","data_revision":"...","kg_revision":"..."}
}
```

