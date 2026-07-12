"""Stratified random subset of MIRAGE for the midterm run (fixed seed ->
reproducible). Sizes chosen with Codex: enough for a ~±3% accuracy CI without
burning quota on all 7663.

Run: python scripts/make_subset.py
"""
import json
import random
from pathlib import Path

from medgraphrag.data.mirage_loader import load_mirage

SIZES = {"medqa": 200, "medmcqa": 300, "pubmedqa": 150, "bioasq": 150, "mmlu": 100}
SEED = 20260712
OUT = "data/midterm_subset.json"


def main():
    rng = random.Random(SEED)
    picked = []
    for subtask, n in SIZES.items():
        qs = load_mirage("data/raw/mirage_benchmark.json", subtask=subtask)
        if len(qs) < n:
            raise ValueError(f"{subtask}: only {len(qs)} < requested {n}")
        sample = rng.sample(qs, n)
        for q in sample:
            picked.append({
                "qid": q.qid,
                "subtask": subtask,
                "question": q.text,
                "options": q.options,
                "answer": q.answer,
            })
        print(f"{subtask}: sampled {n}")
    Path("data").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal {len(picked)} questions -> {OUT} (seed={SEED})")


if __name__ == "__main__":
    main()
