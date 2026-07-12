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
};

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
) {
  const response = await fetch("/api/v1/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, pipeline_id: pipeline }),
    signal,
  });
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`);
  const job = await response.json() as { id: string };
  await new Promise<void>((resolve, reject) => {
    const events = new EventSource(`/api/v1/questions/${job.id}/events`);
    const close = () => events.close();
    signal.addEventListener("abort", () => { close(); reject(new DOMException("Cancelled", "AbortError")); }, { once: true });
    events.addEventListener("status", (event) => {
      const current = JSON.parse((event as MessageEvent).data) as {
        status: string;
        error?: string;
        result?: { answer?: string; citations?: Citation[]; details?: Details };
      };
      if (current.status === "completed") {
        onEvent({ type: "token", text: current.result?.answer ?? "" });
        onEvent({ type: "citations", citations: current.result?.citations ?? [] });
        if (current.result?.details) onEvent({ type: "details", details: current.result.details });
        onEvent({ type: "done" }); close(); resolve();
      } else if (current.status === "failed" || current.status === "cancelled") {
        close(); reject(new Error(current.error || `Request ${current.status}`));
      }
    });
    events.onerror = () => { close(); reject(new Error("The answer stream was interrupted.")); };
  });
}
