export type Citation = {
  pmid: string;
  title: string;
  snippet: string;
  url: string;
};

export type Answer = {
  predicted_type: "yesno" | "factoid" | "list" | "summary";
  exact_answer: string | string[] | null;
  ideal_answer: string;
  citations: Citation[];
  abstained: boolean;
  evidence_support_score: number | null;
  trace_id: string;
  pipeline_id: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number | null;
  model: string;
  attempts: number;
};

const API = import.meta.env.VITE_API_URL ?? "/api";

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function ask(question: string, pipeline_id: string): Promise<Answer> {
  return fetch(`${API}/v1/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, pipeline_id }),
  }).then(checked<Answer>);
}

export function compare(question: string, pipeline_ids: string[]): Promise<Answer[]> {
  return fetch(`${API}/v1/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, pipeline_ids }),
  })
    .then(checked<{ answers: Answer[] }>)
    .then((value) => value.answers);
}
