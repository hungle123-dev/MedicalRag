import { FormEvent, useState } from "react";
import { ask, compare, type Answer } from "./api";
import { AnswerCard } from "./components/AnswerCard";

const PIPELINES = [
  { id: "bm25_rag", name: "BM25 · baseline" },
  { id: "bm25_mesh_rag", name: "BM25 · MeSH expansion" },
];

export default function App() {
  const [question, setQuestion] = useState("What is the role of BRCA1 mutations in breast cancer risk?");
  const [pipeline, setPipeline] = useState(PIPELINES[0].id);
  const [comparison, setComparison] = useState(false);
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError("");
    setAnswers([]);
    try {
      setAnswers(
        comparison
          ? await compare(question.trim(), PIPELINES.map((item) => item.id))
          : [await ask(question.trim(), pipeline)],
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The request could not be completed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="MedicalRAG home">
          <span className="brand-mark">M</span>
          <span>MedicalRAG <b>Lab</b></span>
        </a>
        <span className="status"><i /> Closed-corpus BioASQ research</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="eyebrow">BIOMEDICAL EVIDENCE, MADE INSPECTABLE</span>
          <h1>Ask the literature.<br /><em>Verify the evidence.</em></h1>
          <p>Controlled retrieval experiments and an evidence-grounded QA product share the same frozen pipeline.</p>
        </div>

        <form className="query-panel" onSubmit={submit}>
          <label htmlFor="question">Biomedical question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={2000}
            rows={4}
            placeholder="Ask a biomedical research question…"
          />
          <div className="controls">
            <label className="select-wrap">
              <span>Pipeline</span>
              <select value={pipeline} onChange={(event) => setPipeline(event.target.value)} disabled={comparison}>
                {PIPELINES.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
              </select>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={comparison} onChange={(event) => setComparison(event.target.checked)} />
              <span /> Compare A/B
            </label>
            <button type="submit" disabled={loading || question.trim().length < 3}>
              {loading ? "Searching…" : "Run pipeline →"}
            </button>
          </div>
        </form>
      </section>

      <section className={`answers ${answers.length === 2 ? "two-up" : ""}`} aria-live="polite">
        {loading && <div className="loading"><i /><span>Retrieving and grounding an answer…</span></div>}
        {error && <div className="error" role="alert"><strong>Pipeline error</strong><span>{error}</span></div>}
        {answers.map((answer) => <AnswerCard answer={answer} key={answer.trace_id} />)}
      </section>

      <footer>
        <strong>Research use only.</strong> This system is not medical advice and is not validated for clinical decisions.
      </footer>
    </main>
  );
}
