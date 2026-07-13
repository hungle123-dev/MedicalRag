"""Small frozen-dev generator calibration; never use locked test labels here."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.env import load_dotenv
from app.generator import GatewayGenerator, validate_citations
from app.judge import GatewayJudge, correctness_input, faithfulness_input
from app.pipelines import evidence_budget, text_evidence


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["qwen3.5-397b-a17b", "deepseek-v3.2"])
    args = parser.parse_args()
    load_dotenv(ROOT)
    frozen = json.loads((ROOT / "data/manifests/bioasq_dev_question_ids.json").read_text(encoding="utf-8"))
    ids = frozen["ids"][:args.questions]
    with (ROOT / "data/raw/bioasq/dev.jsonl").open(encoding="utf-8") as stream:
        questions = {row["question_id"]: row for line in stream if (row := json.loads(line)) and row["question_id"] in ids}
    judge = GatewayJudge(ROOT)
    records = []
    for question_id in ids:
        row = questions[question_id]
        evidence, _ = evidence_budget(text_evidence(row["question"], "hybrid"), [])
        registry = {item["id"]: item for item in evidence}
        question_index = ids.index(question_id)
        model_order = args.models[question_index % len(args.models):] + args.models[:question_index % len(args.models)]
        for model in model_order:
            started = time.perf_counter()
            generation = GatewayGenerator(ROOT, model=model).generate(row["question"], evidence)
            latency = round((time.perf_counter() - started) * 1000)
            integrity = validate_citations(generation.answer, evidence)
            cited = [registry[item_id] for item_id in integrity["valid_ids"]]
            correctness = judge.evaluate(correctness_input(row["question"], row["answer"], generation.answer, evidence))
            faithfulness = judge.evaluate(faithfulness_input(generation.answer, cited))
            records.append({"question_id": question_id, "model": model, "latency_ms": latency,
                            "cached": generation.cached,
                            "citation_integrity": integrity, "correctness": correctness,
                            "faithfulness": faithfulness, "answer": generation.answer})
    aggregates = {}
    for model in args.models:
        subset = [record for record in records if record["model"] == model]
        correctness = [record["correctness"] for record in subset]
        faithfulness = [record["faithfulness"] for record in subset]
        uncached = [record["latency_ms"] for record in subset if not record["cached"]]
        aggregates[model] = {
            "citation_integrity_rate": round(sum(record["citation_integrity"]["valid"] for record in subset) / len(subset), 4),
            "correctness_mean_0_2": mean(correctness, "correctness"),
            "completeness_mean_0_2": mean(correctness, "completeness"),
            "unsupported_claim_rate_mean": mean(faithfulness, "unsupported_claim_rate"),
            "uncached_latency_ms_mean": round(sum(uncached) / len(uncached), 1) if uncached else None,
        }
    eligible = [model for model in args.models if aggregates[model]["citation_integrity_rate"] >= 0.95]
    best_quality = max(aggregates[model]["correctness_mean_0_2"] + aggregates[model]["completeness_mean_0_2"]
                       for model in eligible)
    equivalent = [model for model in eligible
                  if best_quality - (aggregates[model]["correctness_mean_0_2"] +
                                     aggregates[model]["completeness_mean_0_2"]) <= 0.1]
    ranking = sorted(equivalent, key=lambda model: (
        aggregates[model]["unsupported_claim_rate_mean"],
        aggregates[model]["uncached_latency_ms_mean"] or float("inf"), model))
    result = {"track": "exploratory_frozen_dev_generator_calibration", "question_ids": ids,
              "models": args.models, "judge_model": judge.model,
              "selection_rule": "citation integrity >=0.95; quality within 0.1 is equivalent; then unsupported claims and uncached latency",
              "aggregates": aggregates, "selected": ranking[0],
              "warning": "Small LLM-judge calibration is model selection evidence, not a medical performance claim."}
    directory = ROOT / "artifacts/experiments/generator_calibration"; directory.mkdir(parents=True, exist_ok=True)
    (directory / f"dev_{args.questions}_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    target = ROOT / f"data/manifests/generator_calibration_dev_{args.questions}_20260713.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
