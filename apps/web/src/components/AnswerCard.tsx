import type { Answer } from "../api";

function exactText(answer: Answer): string {
  if (answer.exact_answer === null) return "Not applicable for summary answers";
  return Array.isArray(answer.exact_answer) ? answer.exact_answer.join(" · ") : answer.exact_answer;
}

export function AnswerCard({ answer }: { answer: Answer }) {
  return (
    <article className="answer-card" aria-label={`Answer from ${answer.pipeline_id}`}>
      <div className="answer-head">
        <div>
          <span className="eyebrow">{answer.pipeline_id.replaceAll("_", " ")}</span>
          <h2>{answer.abstained ? "Insufficient evidence" : "Evidence-grounded answer"}</h2>
        </div>
        <span className="type-pill">{answer.predicted_type}</span>
      </div>

      <section className="exact-block">
        <span>Exact answer</span>
        <strong>{exactText(answer)}</strong>
      </section>
      <p className="ideal-answer">{answer.ideal_answer}</p>

      <div className="metrics" aria-label="Runtime metadata">
        <span><strong>{(answer.latency_ms / 1000).toFixed(1)}s</strong> latency</span>
        <span><strong>{answer.input_tokens + answer.output_tokens}</strong> tokens</span>
        <span><strong>{answer.evidence_support_score?.toFixed(2) ?? "—"}</strong> support</span>
        <span><strong>{answer.estimated_cost_usd === null ? "unknown" : `$${answer.estimated_cost_usd.toFixed(4)}`}</strong> cost</span>
        <span><strong>{answer.model || "—"}</strong> model</span>
      </div>

      <details className="evidence">
        <summary>Inspect {answer.citations.length} evidence citations</summary>
        {answer.citations.length === 0 ? (
          <p className="empty">No validated PMID was cited.</p>
        ) : (
          answer.citations.map((citation) => (
            <div className="citation" key={citation.pmid}>
              <a href={citation.url} target="_blank" rel="noreferrer">
                PMID {citation.pmid} ↗
              </a>
              <h3>{citation.title}</h3>
              <p>{citation.snippet}</p>
            </div>
          ))
        )}
      </details>
      <small className="trace">Trace {answer.trace_id}</small>
    </article>
  );
}
