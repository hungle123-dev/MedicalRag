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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
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


def _call_with_retry(llm_box, q, context, retries=6, wall_timeout=25):
    """Wall-clock timeout via a worker thread: the SDK's own timeout doesn't
    reliably cut a socket that wedges through the Cloudflare proxy, so we
    abandon the future and retry with a FRESH client. A ONE-SHOT executor per
    call (not a shared pool) — a shared pool's workers get permanently consumed
    by abandoned/hung threads until it's exhausted and every future queues
    forever. A throwaway 1-worker pool costs a bit of overhead but never clogs.
    llm_box is a 1-element list so we can swap the client on timeout."""
    for attempt in range(retries):
        llm = llm_box[0]
        exec_ = ThreadPoolExecutor(max_workers=1)
        fut = exec_.submit(llm.choose, q["question"], q["options"], context)
        try:
            result = fut.result(timeout=wall_timeout)
            exec_.shutdown(wait=False)  # non-blocking: don't wait on a done thread
            return result
        except FutureTimeout:
            exec_.shutdown(wait=False)  # abandon the hung thread, don't block on it
            if attempt == retries - 1:
                raise RuntimeError(f"wall-timeout after {retries} tries")
            llm_box[0] = OpenAICompatLLM(llm._model)  # fresh connection pool
        except Exception:
            exec_.shutdown(wait=False)
            if attempt == retries - 1:
                raise
            time.sleep(min(15, 2 * (attempt + 1)))
    return None


def run_cell(model: str, arm: str, questions: list, retriever, limit: int | None):
    out_path = RESULTS_DIR / f"{_safe(model)}_{arm}.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    done = _load_done(out_path)
    qs = questions[:limit] if limit else questions
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[{model} | {arm}] {len(done)} done, {len(todo)} to run")

    llm_box = [OpenAICompatLLM(model)]
    n_fail = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, q in enumerate(todo, 1):
            # bm25s query is ~150ms, so live retrieval is fine (no cache needed)
            evidence = retriever.retrieve(q["question"], K) if arm == "E1" else []
            context = "\n".join(e.content for e in evidence)
            try:
                choice = _call_with_retry(llm_box, q, context, retries=6)
            except Exception as e:
                # log & skip one bad question rather than kill an 1800-call run;
                # a rerun (resume) will retry skipped qids since they're not written
                n_fail += 1
                print(f"  SKIP {q['qid']}: {str(e)[:80]}")
                continue
            answer_text = q["options"][q["answer"]].lower()
            row = {
                "qid": q["qid"],
                "subtask": q["subtask"],
                "predicted": choice,
                "gold": q["answer"],
                "correct": choice == q["answer"],
                "n_evidence": len(evidence),
                "evidence_sources": [e.source for e in evidence],
                "answer_in_context": any(answer_text in e.content.lower() for e in evidence),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {model}|{arm}: {i}/{len(todo)}")
    print(f"  done -> {out_path}" + (f"  ({n_fail} skipped)" if n_fail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per cell (smoke test)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    args = ap.parse_args()

    load_env()
    questions = json.loads(Path(SUBSET).read_text(encoding="utf-8"))
    retriever = None
    if "E1" in args.arms:
        with open(INDEX, "rb") as f:
            retriever = pickle.load(f)

    for model in args.models:
        for arm in args.arms:
            run_cell(model, arm, questions, retriever, args.limit)


if __name__ == "__main__":
    main()
