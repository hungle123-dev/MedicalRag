export type Citation = {
  id: string;
  title?: string;
  url?: string;
  snippet?: string;
  type?: "text" | "graph";
};

export type Details = {
  latency_ms?: number;
  pipeline?: string;
  graph_paths?: string[];
  degraded?: boolean;
  degraded_reason?: string;
  linked_entities?: { name: string; type: string; confidence: number }[];
  budget?: { token_budget?: number; graph_tokens_actual?: number; text_tokens_actual?: number; evidence_items?: number };
  generator?: { provider: string; model: string; cached: boolean };
};

export type Result = { answer: string; citations: Citation[]; evidence: Citation[]; details: Details };
export type Readiness = { status: string; pipelines: Record<string, boolean>; dependencies: Record<string, unknown> };

export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "details"; details: Details }
  | { type: "done" }
  | { type: "error"; message: string };

export async function streamAnswer(
  question: string,
  pipeline: string,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
): Promise<Result> {
  const response = await fetch("/api/v1/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, pipeline_id: pipeline }),
    signal,
  });
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`);
  const job = await response.json() as { id: string };
  return await new Promise<Result>((resolve, reject) => {
    const events = new EventSource(`/api/v1/questions/${job.id}/events`);
    const close = () => events.close();
    signal.addEventListener("abort", () => {
      void fetch(`/api/v1/questions/${job.id}`, { method: "DELETE", keepalive: true });
      close(); reject(new DOMException("Cancelled", "AbortError"));
    }, { once: true });
    events.addEventListener("status", (event) => {
      const current = JSON.parse((event as MessageEvent).data) as {
        status: string;
        error?: string;
        result?: Partial<Result>;
      };
      if (current.status === "completed") {
        const result: Result = { answer: current.result?.answer ?? "", citations: current.result?.citations ?? [],
          evidence: current.result?.evidence ?? [], details: current.result?.details ?? {} };
        onEvent({ type: "token", text: result.answer });
        onEvent({ type: "citations", citations: result.citations });
        onEvent({ type: "details", details: result.details });
        onEvent({ type: "done" }); close(); resolve(result);
      } else if (current.status === "failed" || current.status === "cancelled") {
        close(); reject(new Error(current.error || `Request ${current.status}`));
      }
    });
    events.onerror = () => { close(); reject(new Error("The answer stream was interrupted.")); };
  });
}

export function safePubMedUrl(value?: string): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "pubmed.ncbi.nlm.nih.gov" ? url.href : undefined;
  } catch { return undefined; }
}

export async function fetchReadiness(): Promise<Readiness> {
  const response = await fetch("/api/v1/ready");
  return await response.json() as Readiness;
}
