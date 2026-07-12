"""End-to-end demo: real MIRAGE questions + real MedRAG textbook corpus ->
E0 (no retrieval) vs E1 (BM25) -> accuracy.

Uses MockLLM by default (no network needed, runs anywhere). Swap --llm real
to use OpenAI/Gemini once network/keys are available on the machine running
this (see llm/openai_client.py, llm/gemini.py — same interface, drop-in).

Run: python scripts/run_demo.py --n 30 --subtask medqa
"""
import argparse
import json

from medgraphrag.data.mirage_loader import load_mirage
from medgraphrag.data.corpus_loader import load_textbook_corpus
from medgraphrag.pipeline.arms import build_arm
from medgraphrag.pipeline.runner import run_arm
from medgraphrag.eval.accuracy import accuracy
from medgraphrag.llm.mock import MockLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of questions to sample")
    ap.add_argument("--subtask", default="medqa", choices=["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"])
    ap.add_argument("--corpus-limit", type=int, default=5000, help="chunks to load for BM25")
    args = ap.parse_args()

    questions = load_mirage("data/raw/mirage_benchmark.json", subtask=args.subtask)[: args.n]
    corpus = load_textbook_corpus("data/raw/medrag_textbooks/chunk", limit=args.corpus_limit)
    print(f"Loaded {len(questions)} real '{args.subtask}' questions, "
          f"{len(corpus)} real textbook chunks as corpus.\n")

    llm = MockLLM(default="A")
    results = {}
    for arm_name in ("E0", "E1"):
        arm = build_arm(arm_name, corpus, llm, k=5)
        res = run_arm(arm_name, arm, questions)
        acc = accuracy(res.predictions, questions)
        results[arm_name] = acc
        print(f"{arm_name}: accuracy={acc:.3f}  (n={len(questions)})")
        # show one real sample prediction as an input/output example
        p0 = res.predictions[0]
        q0 = questions[0]
        print(f"  sample -> qid={p0.qid} predicted={p0.choice} gold={q0.answer} "
              f"evidence_count={len(p0.evidence)}")

    print("\nNOTE: MockLLM used (no network available in this environment).")
    print("These accuracy numbers are NOT the real reported results — they only")
    print("prove the pipeline wiring is correct. Re-run with a real LLM client")
    print("(llm/openai_client.py or llm/gemini.py) on a machine with API access")
    print("for real accuracy numbers.")

    with open("data/demo_run_result.json", "w", encoding="utf-8") as f:
        json.dump({"subtask": args.subtask, "n": len(questions), "accuracy": results,
                   "llm": "mock"}, f, indent=2)
    print("\nSaved: data/demo_run_result.json")


if __name__ == "__main__":
    main()
