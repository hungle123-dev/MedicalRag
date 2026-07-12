"""Runs E0 (no retrieval) and E1 (BM25 RAG) over the 900-question subset with
real LLMs through the proxy. Resumable: each (model,arm) writes per-qid rows to
results/<model>_<arm>.jsonl; a rerun skips qids already present.

Also records the proxy retrieval signal for E1: does any retrieved chunk contain
the gold option's text (answer_in_context) — an honest stand-in for Recall@k
since MIRAGE has no evidence labels.

Run:
  python scripts/run_experiment.py --limit 50            # smoke test
  python scripts/run_experiment.py                       # full 900 x arms x models
"""
import argparse
import json
import pickle
import time
from pathlib import Path

from medgraphrag.config import load_env
from medgraphrag.llm.openai_client import OpenAICompatLLM
from medgraphrag.retrieval.null import NullRetriever

MODELS = ["gpt-4.1-nano", "gemini-2.5-flash-lite"]
ARMS = ["E0", "E1"]
K = 5
SUBSET = "data/midterm_subset.json"
INDEX = "data/bm25_index.pkl"
RESULTS_DIR = Path("results")


def _safe(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["qid"])
    return done


def _answer_in_context(option_text: str, evidence) -> bool:
    t = option_text.lower()
    return any(t in e.content.lower() for e in evidence)


def _call_with_retry(llm, q, context, retries=3):
    for attempt in range(retries):
        try:
            return llm.choose(q["question"], q["options"], context)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def run_cell(model: str, arm: str, questions: list, retriever, limit: int | None):
    out_path = RESULTS_DIR / f"{_safe(model)}_{arm}.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    done = _load_done(out_path)
    qs = questions[:limit] if limit else questions
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[{model} | {arm}] {len(done)} done, {len(todo)} to run")

    llm = OpenAICompatLLM(model)
    with out_path.open("a", encoding="utf-8") as fh:
        for i, q in enumerate(todo, 1):
            if arm == "E1":
                evidence = retriever.retrieve(q["question"], K)
            else:
                evidence = []
            context = "\n".join(e.content for e in evidence)
            choice = _call_with_retry(llm, q, context)
            row = {
                "qid": q["qid"],
                "subtask": q["subtask"],
                "predicted": choice,
                "gold": q["answer"],
                "correct": choice == q["answer"],
                "n_evidence": len(evidence),
                "evidence_sources": [e.source for e in evidence],
                "answer_in_context": _answer_in_context(q["options"][q["answer"]], evidence),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {model}|{arm}: {i}/{len(todo)}")
    print(f"  done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per cell (smoke test)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    args = ap.parse_args()

    load_env()
    questions = json.loads(Path(SUBSET).read_text(encoding="utf-8"))
    with open(INDEX, "rb") as f:
        retriever = pickle.load(f)

    for model in args.models:
        for arm in args.arms:
            run_cell(model, arm, questions, retriever, args.limit)


if __name__ == "__main__":
    main()
