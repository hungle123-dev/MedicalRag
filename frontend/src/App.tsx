import { FormEvent, useRef, useState } from "react";
import { Citation, Details, streamAnswer } from "./api";

const pipelines = [
  ["B0", "Closed-book LLM"],
  ["B1", "BM25 RAG"],
  ["B2", "MedCPT RAG"],
  ["B3", "Hybrid text RAG"],
  ["G1", "PrimeKG only"],
  ["G2", "Text + PrimeKG"],
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [pipeline, setPipeline] = useState("G2");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [details, setDetails] = useState<Details>({});
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || status === "loading") return;
    controller.current = new AbortController();
    setAnswer("");
    setCitations([]);
    setDetails({});
    setError("");
    setStatus("loading");
    try {
      await streamAnswer(cleanQuestion, pipeline, controller.current.signal, (message) => {
        if (message.type === "token") setAnswer((current) => current + message.text);
        if (message.type === "citations") setCitations(message.citations);
        if (message.type === "details") setDetails(message.details);
        if (message.type === "error") throw new Error(message.message);
      });
      setStatus("done");
    } catch (reason) {
      if ((reason as Error).name === "AbortError") return setStatus("idle");
      setError(reason instanceof Error ? reason.message : "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">Evidence-grounded research demo</p>
        <h1>Medical RAG</h1>
        <p className="lede">Compare text and knowledge-graph retrieval on the same medical question.</p>
      </header>

      <aside className="warning" role="note">
        <strong>Research use only.</strong> Answers may be incomplete or incorrect and are not medical advice. Seek a qualified clinician for care decisions.
      </aside>

      <form onSubmit={submit}>
        <label htmlFor="question">Medical question</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Can propranolol worsen asthma, and why?"
          maxLength={4000}
          rows={4}
          required
        />
        <div className="controls">
          <label htmlFor="pipeline">Pipeline</label>
          <select id="pipeline" value={pipeline} onChange={(event) => setPipeline(event.target.value)}>
            {pipelines.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
          <button type="submit" disabled={status === "loading" || !question.trim()}>
            {status === "loading" ? "Retrieving…" : "Ask"}
          </button>
          {status === "loading" && <button className="secondary" type="button" onClick={() => controller.current?.abort()}>Stop</button>}
        </div>
      </form>

      <section className="answer" aria-live="polite" aria-busy={status === "loading"}>
        <h2>Answer</h2>
        {status === "idle" && !answer && <p className="muted">Your grounded answer will appear here.</p>}
        {answer && <p className="answer-text">{answer}</p>}
        {status === "loading" && <span className="status">Searching evidence and generating an answer…</span>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {citations.length > 0 && (
        <section>
          <h2>Evidence</h2>
          <ol className="citations">
            {citations.map((citation) => (
              <li key={citation.id}>
                <span className="badge">{citation.type === "graph" ? "Graph" : "Text"}</span>
                {citation.url ? <a href={citation.url} target="_blank" rel="noreferrer">{citation.title || citation.id}</a> : <strong>{citation.title || citation.id}</strong>}
                {citation.snippet && <p>{citation.snippet}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {(details.latency_ms !== undefined || details.graph_paths?.length) && (
        <details>
          <summary>Retrieval details</summary>
          {details.latency_ms !== undefined && <p>Latency: {details.latency_ms.toLocaleString()} ms</p>}
          {details.graph_paths?.map((path) => <code key={path}>{path}</code>)}
        </details>
      )}
    </main>
  );
}
