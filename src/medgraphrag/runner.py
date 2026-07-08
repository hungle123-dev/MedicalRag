"""Config-driven run: resolve a config dict -> run arm -> RunResult -> disk.

A config is: {arm: "E1", dataset: "fixture", k: 3, llm: {default: "A", rules: {}}}.
Everything reproducible from that dict; results are written next to a copy of
the config under experiments/.
"""
import json
import os
from dataclasses import dataclass

import medgraphrag.arms.standard  # noqa: F401  (registers E0/E1/E2)
from medgraphrag.core.registry import build
from medgraphrag.core.types import Question, Prediction
from medgraphrag.data.loaders import load_dataset
from medgraphrag.eval.accuracy import accuracy
from medgraphrag.llm.mock import MockLLM


@dataclass
class RunResult:
    arm_name: str
    dataset: str
    accuracy: float
    predictions: list[Prediction]


def _make_llm(spec: dict):
    # ponytail: only MockLLM exists; real API adapter selected by spec["kind"] later
    return MockLLM(rules=spec.get("rules"), default=spec.get("default", "A"))


def run_config(config: dict) -> RunResult:
    questions, corpus, triples = load_dataset(config["dataset"])
    ctx = {
        "corpus": corpus,
        "triples": triples,
        "llm": _make_llm(config.get("llm", {})),
        "k": config.get("k", 3),
    }
    arm = build(config["arm"], ctx)
    preds = [arm.answer(q) for q in questions]
    acc = accuracy(preds, questions)
    return RunResult(config["arm"], config["dataset"], acc, preds)


def save_results(results: list[RunResult], path: str, config: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": config,
        "runs": [
            {"arm": r.arm_name, "dataset": r.dataset,
             "accuracy": r.accuracy, "n": len(r.predictions)}
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
