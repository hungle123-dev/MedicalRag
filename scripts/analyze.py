"""Analyzes results/<model>_<arm>.jsonl -> accuracy tables, E0-vs-E1 delta,
McNemar paired test, answer_in_context rate, and case studies. Writes
results/analysis.json + prints tables. Only reads what actually ran (robust to
partial runs).

Run: python scripts/analyze.py
"""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

RESULTS_DIR = Path("results")
MODELS = ["gpt-4.1-nano", "gemini-2.5-flash-lite"]
ARMS = ["E0", "E1"]


def _safe(model):
    return model.replace("/", "_").replace(".", "-")


def _load(model, arm):
    p = RESULTS_DIR / f"{_safe(model)}_{arm}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def mcnemar(e0: dict, e1: dict):
    """Paired: over qids both arms answered, count where they disagree."""
    shared = set(e0) & set(e1)
    e0_only = sum(1 for q in shared if e0[q]["correct"] and not e1[q]["correct"])
    e1_only = sum(1 for q in shared if e1[q]["correct"] and not e0[q]["correct"])
    n = e0_only + e1_only
    if n == 0:
        return {"e0_only_correct": 0, "e1_only_correct": 0, "p_value": 1.0, "n_shared": len(shared)}
    # exact McNemar = binomial test on the discordant pairs
    p = binomtest(e1_only, n, 0.5).pvalue
    return {"e0_only_correct": e0_only, "e1_only_correct": e1_only,
            "p_value": round(p, 4), "n_shared": len(shared)}


def analyze_model(model: str) -> dict:
    e0 = _load(model, "E0")
    e1 = _load(model, "E1")
    if not e0 and not e1:
        return {}
    report = {"overall": {}, "by_subtask": {}}

    for arm, data in (("E0", e0), ("E1", e1)):
        report["overall"][arm] = {"n": len(data), "accuracy": round(_acc(list(data.values())), 4)}
    # per-subtask
    subtasks = {r["subtask"] for r in list(e0.values()) + list(e1.values())}
    for st in sorted(subtasks):
        report["by_subtask"][st] = {}
        for arm, data in (("E0", e0), ("E1", e1)):
            rows = [r for r in data.values() if r["subtask"] == st]
            report["by_subtask"][st][arm] = round(_acc(rows), 4) if rows else None
    # E1 retrieval signal
    if e1:
        report["overall"]["E1_answer_in_context_rate"] = round(
            sum(r["answer_in_context"] for r in e1.values()) / len(e1), 4)
    # paired significance
    report["mcnemar_E0_vs_E1"] = mcnemar(e0, e1)
    if e0 and e1:
        report["overall"]["delta_E1_minus_E0"] = round(
            report["overall"]["E1"]["accuracy"] - report["overall"]["E0"]["accuracy"], 4)
    return report


def case_studies(model: str, n=5) -> dict:
    e0, e1 = _load(model, "E0"), _load(model, "E1")
    shared = set(e0) & set(e1)
    buckets = {"e1_fixed_e0": [], "e1_broke_e0": [], "retrieved_but_wrong": []}
    for q in shared:
        a0, a1 = e0[q], e1[q]
        if a1["correct"] and not a0["correct"]:
            buckets["e1_fixed_e0"].append(q)
        if a0["correct"] and not a1["correct"]:
            buckets["e1_broke_e0"].append(q)
        if a1["answer_in_context"] and not a1["correct"]:
            buckets["retrieved_but_wrong"].append(q)
    return {k: v[:n] for k, v in buckets.items()}


def main():
    out = {}
    for model in MODELS:
        rep = analyze_model(model)
        if not rep:
            print(f"[{model}] no results yet, skipping")
            continue
        out[model] = {"metrics": rep, "case_study_qids": case_studies(model)}
        print(f"\n{'='*60}\n{model}\n{'='*60}")
        ov = rep["overall"]
        print(f"  E0 acc={ov.get('E0',{}).get('accuracy')} (n={ov.get('E0',{}).get('n')})")
        print(f"  E1 acc={ov.get('E1',{}).get('accuracy')} (n={ov.get('E1',{}).get('n')})")
        print(f"  delta (E1-E0)={ov.get('delta_E1_minus_E0')}")
        print(f"  E1 answer_in_context={ov.get('E1_answer_in_context_rate')}")
        print(f"  McNemar: {rep['mcnemar_E0_vs_E1']}")
        print("  by subtask (E0 / E1):")
        for st, v in rep["by_subtask"].items():
            print(f"    {st:10s} {v['E0']} / {v['E1']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
