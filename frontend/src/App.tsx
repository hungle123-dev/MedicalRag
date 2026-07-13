import { FormEvent, useRef, useState } from "react";
import { Citation, Details, Result, safePubMedUrl, streamAnswer } from "./api";

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
  const [evidence, setEvidence] = useState<Citation[]>([]);
  const [compare, setCompare] = useState(false);
  const [comparison, setComparison] = useState<Record<string, Result>>({});
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
    setEvidence([]);
    setComparison({});
    setDetails({});
    setError("");
    setStatus("loading");
    try {
      if (compare) {
        const [b3, g2] = await Promise.all([
          streamAnswer(cleanQuestion, "B3", controller.current.signal, () => undefined),
          streamAnswer(cleanQuestion, "G2", controller.current.signal, () => undefined),
        ]);
        setComparison({ B3: b3, G2: g2 });
        setStatus("done");
        return;
      }
      const result = await streamAnswer(cleanQuestion, pipeline, controller.current.signal, (message) => {
        if (message.type === "token") setAnswer((current) => current + message.text);
        if (message.type === "citations") setCitations(message.citations);
        if (message.type === "details") setDetails(message.details);
        if (message.type === "error") throw new Error(message.message);
      });
      setEvidence(result.evidence);
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
        <strong>Research use only.</strong> Answers may be incomplete or incorrect and are not medical advice. Do not enter identifiable patient data. Seek a qualified clinician for care decisions.
      </aside>

      <form onSubmit={submit}>
        <label htmlFor="question">Medical question</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Can propranolol worsen asthma, and why?"
          maxLength={2000}
          rows={4}
          required
        />
        <div className="controls">
          <label htmlFor="pipeline">Pipeline</label>
          <select id="pipeline" value={pipeline} onChange={(event) => setPipeline(event.target.value)}>
            {pipelines.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
          <label className="compare-toggle"><input type="checkbox" checked={compare} onChange={(event) => setCompare(event.target.checked)} /> Compare B3/G2</label>
          <button type="submit" disabled={status === "loading" || !question.trim()}>
            {status === "loading" ? "Retrieving…" : "Ask"}
          </button>
          {status === "loading" && <button className="secondary" type="button" onClick={() => controller.current?.abort()}>Stop</button>}
        </div>
      </form>

      {Object.keys(comparison).length > 0 && <section aria-live="polite">
        <h2>B3/G2 matched comparison</h2>
        <div className="comparison">
          {(["B3", "G2"] as const).map((id) => <article key={id}>
            <h3>{id}: {id === "B3" ? "Text RAG" : "Text + PrimeKG"}</h3>
            <p className="answer-text">{comparison[id]?.answer}</p>
            <small>{comparison[id]?.details.latency_ms?.toLocaleString()} ms · {comparison[id]?.evidence.length} evidence items</small>
          </article>)}
        </div>
      </section>}

      <section className="answer" aria-live="polite" aria-busy={status === "loading"}>
        <h2>Answer</h2>
        {status === "idle" && !answer && <p className="muted">Your grounded answer will appear here.</p>}
        {answer && <p className="answer-text">{answer}</p>}
        {status === "loading" && <span className="status">Searching evidence and generating an answer…</span>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {(evidence.length > 0 || citations.length > 0) && (
        <section>
          <h2>PubMed evidence and citations</h2>
          <ol className="citations">
            {evidence.filter((item) => item.type === "text").map((citation) => (
              <li key={citation.id}>
                <span className="badge">Text</span>
                {citations.some((item) => item.id === citation.id) && <span className="badge cited">Cited</span>}
                {safePubMedUrl(citation.url) ? <a href={safePubMedUrl(citation.url)} target="_blank" rel="noreferrer">{citation.title || citation.id}</a> : <strong>{citation.title || citation.id}</strong>}
                {citation.snippet && <p>{citation.snippet}</p>}
              </li>
            ))}
          </ol>
          <h2 className="subheading">PrimeKG paths</h2>
          {evidence.some((item) => item.type === "graph") ? <ol className="citations">
            {evidence.filter((item) => item.type === "graph").map((item) => <li key={item.id}><span className="badge">Graph</span>{citations.some((citation) => citation.id === item.id) && <span className="badge cited">Cited</span>}<strong>{item.id}</strong><p>{item.snippet}</p></li>)}
          </ol> : <p className="muted">No graph evidence passed the retrieval and budget filters.</p>}
        </section>
      )}

      {(details.latency_ms !== undefined || details.graph_paths?.length) && (
        <details>
          <summary>Retrieval details</summary>
          {details.degraded && <p className="error">Degraded pipeline: {details.degraded_reason}</p>}
          {details.latency_ms !== undefined && <p>Latency: {details.latency_ms.toLocaleString()} ms</p>}
          {details.generator && <p>Generator: {details.generator.provider}/{details.generator.model}{details.generator.cached ? " (cached)" : ""}</p>}
          {details.budget && <p>Evidence budget: text {details.budget.text_tokens_actual ?? 0}, graph {details.budget.graph_tokens_actual ?? 0} / {details.budget.token_budget ?? 1800} tokens.</p>}
          {details.linked_entities?.length && <p>Linked entities: {details.linked_entities.map((entity) => `${entity.name} (${entity.type})`).join(", ")}</p>}
          {details.graph_paths?.map((path) => <code key={path}>{path}</code>)}
        </details>
      )}
    </main>
  );
}
