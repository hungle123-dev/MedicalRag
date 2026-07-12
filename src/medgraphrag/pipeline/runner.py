"""Runs an arm over a list of questions, returns predictions. Pure orchestration
— no I/O beyond what callers pass in, so it works identically with MockLLM
(wiring test) or a real LLM client (accuracy run)."""
from dataclasses import dataclass

from medgraphrag.core.types import Question, Prediction
from medgraphrag.pipeline.arms import Arm


@dataclass
class RunResult:
    arm_name: str
    predictions: list[Prediction]


def run_arm(arm_name: str, arm: Arm, questions: list[Question]) -> RunResult:
    preds = [arm.answer(q) for q in questions]
    return RunResult(arm_name=arm_name, predictions=preds)
