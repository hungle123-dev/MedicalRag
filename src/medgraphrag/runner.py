"""Config-driven run: resolve a config dict -> run arm -> RunResult -> disk.

Config shape:
  arm: "E1"
  dataset: "fixture"
  k: 3
  llm: {default: "A", rules: {}}
  arm_modules: ["medgraphrag.arms.standard"]   # optional; imported so their
                                               # @register(...) side effects run

A new strategy E3 lives in its own module and is enabled purely by listing that
module here — no existing file changes.
"""
import importlib
import json
import os
from dataclasses import dataclass

from medgraphrag.core.registry import build
from medgraphrag.core.types import Prediction
from medgraphrag.data.loaders import load_dataset
from medgraphrag.eval.accuracy import accuracy
from medgraphrag.eval.retrieval import recall_at_k, mrr
from medgraphrag.llm.mock import MockLLM

DEFAULT_ARM_MODULES = ["medgraphrag.arms.standard"]


@dataclass
class RunResult:
    arm_name: str
    dataset: str
    accuracy: float
    recall_at_k: float
    mrr: float
    predictions: list[Prediction]


def _make_llm(spec: dict):
    # ponytail: only MockLLM exists; real API adapter selected by spec["kind"] later
    return MockLLM(rules=spec.get("rules"), default=spec.get("default", "A"))


def _load_arm_modules(config: dict) -> None:
    for mod in config.get("arm_modules", DEFAULT_ARM_MODULES):
        importlib.import_module(mod)


def run_config(config: dict) -> RunResult:
    _load_arm_modules(config)
    questions, corpus, triples = load_dataset(config["dataset"])
    ctx = {
        "corpus": corpus,
        "triples": triples,
        "llm": _make_llm(config.get("llm", {})),
        "k": config.get("k", 3),
    }
    arm = build(config["arm"], ctx)
    preds = [arm.answer(q) for q in questions]
    return RunResult(
        arm_name=config["arm"],
        dataset=config["dataset"],
        accuracy=accuracy(preds, questions),
        recall_at_k=recall_at_k(preds, questions),
        mrr=mrr(preds, questions),
        predictions=preds,
    )


def _pred_to_dict(p: Prediction) -> dict:
    return {
        "qid": p.qid,
        "choice": p.choice,
        "evidence": [
            {"source": e.source, "score": e.score, "content": e.content}
            for e in p.evidence
        ],
    }


def save_results(results: list[RunResult], path: str, config: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": config,
        "runs": [
            {
                "arm": r.arm_name,
                "dataset": r.dataset,
                "accuracy": r.accuracy,
                "recall_at_k": r.recall_at_k,
                "mrr": r.mrr,
                "n": len(r.predictions),
                "predictions": [_pred_to_dict(p) for p in r.predictions],
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
