"""EDA on the REAL downloaded data: MIRAGE (7663 Qs, 5 subtasks) + MedRAG
textbook corpus (202MB, 18 books). Prints stats used directly in the midterm
report; no plotting lib dependency (kept simple, numbers only).

Run: python scripts/eda.py
"""
import json
import statistics
from pathlib import Path

from medgraphrag.data.mirage_loader import load_mirage, SUBTASKS
from medgraphrag.data.corpus_loader import load_textbook_corpus


def word_count(text: str) -> int:
    return len(text.split())


def eda_mirage(path: str) -> dict:
    print("=" * 70)
    print("MIRAGE — question set EDA")
    print("=" * 70)
    report = {}
    for st in SUBTASKS:
        qs = load_mirage(path, subtask=st)
        q_lens = [word_count(q.text) for q in qs]
        opt_lens = [word_count(v) for q in qs for v in q.options.values()]
        n_opts = [len(q.options) for q in qs]
        answer_dist = {}
        for q in qs:
            answer_dist[q.answer] = answer_dist.get(q.answer, 0) + 1
        report[st] = {
            "n": len(qs),
            "n_options_typical": statistics.mode(n_opts),
            "question_words_mean": round(statistics.mean(q_lens), 1),
            "question_words_median": statistics.median(q_lens),
            "question_words_max": max(q_lens),
            "option_words_mean": round(statistics.mean(opt_lens), 1),
            "answer_key_distribution": dict(sorted(answer_dist.items())),
        }
        r = report[st]
        print(f"\n[{st}] n={r['n']}  #options(typical)={r['n_options_typical']}")
        print(f"  question length (words): mean={r['question_words_mean']} "
              f"median={r['question_words_median']} max={r['question_words_max']}")
        print(f"  option length (words): mean={r['option_words_mean']}")
        print(f"  answer-key distribution: {r['answer_key_distribution']}")
    total = sum(r["n"] for r in report.values())
    print(f"\nTOTAL questions across 5 subtasks: {total}")
    return report


def eda_corpus(dir_path: str) -> dict:
    print("\n" + "=" * 70)
    print("MedRAG textbooks — corpus EDA")
    print("=" * 70)
    root = Path(dir_path)
    files = sorted(root.glob("*.jsonl"))
    report = {"books": {}}
    total_chunks = 0
    total_words = 0
    for f in files:
        n_chunks = 0
        word_lens = []
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                n_chunks += 1
                word_lens.append(word_count(obj["content"]))
        report["books"][f.stem] = {
            "n_chunks": n_chunks,
            "chunk_words_mean": round(statistics.mean(word_lens), 1) if word_lens else 0,
            "chunk_words_median": statistics.median(word_lens) if word_lens else 0,
        }
        total_chunks += n_chunks
        total_words += sum(word_lens)
        print(f"  {f.stem:28s} chunks={n_chunks:6d}  "
              f"mean_words={report['books'][f.stem]['chunk_words_mean']:6.1f}")
    report["total_chunks"] = total_chunks
    report["total_words"] = total_words
    print(f"\nTOTAL: {len(files)} books, {total_chunks} chunks, {total_words:,} words")
    return report


if __name__ == "__main__":
    mirage_report = eda_mirage("data/raw/mirage_benchmark.json")
    corpus_report = eda_corpus("data/raw/medrag_textbooks/chunk")

    out = {"mirage": mirage_report, "corpus": corpus_report}
    Path("data").mkdir(exist_ok=True)
    Path("data/eda_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved: data/eda_report.json")
