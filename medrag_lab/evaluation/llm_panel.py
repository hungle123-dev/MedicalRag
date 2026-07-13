from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Literal

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

from medrag_lab.settings import ROOT, settings


class DirectJudgment(BaseModel):
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reference_discrepancies: list[str] = Field(default_factory=list)
    justification: str
    correctness: float = Field(ge=0, le=4)
    completeness: float = Field(ge=0, le=4)
    evidence_faithfulness: float = Field(ge=0, le=4)


class PairwiseJudgment(BaseModel):
    evidence_comparison: str
    justification: str
    winner: Literal["A", "B", "tie"]


class LLMPanel:
    def __init__(self, config_path: Path | None = None):
        config = settings()
        self.path = config_path or ROOT / "configs" / "judges" / "panel.yaml"
        self.config = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.client = OpenAI(
            api_key=config.openai_api_key.get_secret_value() if config.openai_api_key else "",
            base_url=config.openai_base_url.rstrip("/"),
            timeout=90,
            max_retries=0,
        )
        self.cache = ROOT / "artifacts" / "cache" / "judges"

    def _json_call(self, model: str, system: str, user: str) -> dict:
        payload = {"model": model, "system": system, "user": user, "temperature": 0}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{digest}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=4_000,
                    **({"response_format": {"type": "json_object"}} if attempt == 1 else {}),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                message = response.choices[0].message
                text = message.content or getattr(message, "reasoning_content", None) or ""
                start, end = text.find("{"), text.rfind("}")
                if start < 0 or end < start:
                    raise ValueError(
                        f"Judge {model} returned no JSON object (content_length={len(text)}, "
                        f"preview={text[:180]!r})"
                    )
                value = json.loads(text[start : end + 1])
                break
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(attempt)
        else:  # pragma: no cover
            raise RuntimeError("Judge retry loop exhausted") from last_error
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return value

    def direct(self, question: str, answer: str, reference: str, evidence: str) -> dict:
        system = """You are an impartial biomedical QA evaluator. Pipeline identity is hidden.
First enumerate evidence-supported and unsupported answer claims and discrepancies with the
reference. Then justify and score correctness, completeness, and evidence faithfulness from
0 (unacceptable) to 4 (fully supported/correct). The reference is an evaluation aid, while
the supplied evidence is the only source for faithfulness. Return JSON only with keys:
supported_claims, unsupported_claims, reference_discrepancies, justification, correctness,
completeness, evidence_faithfulness. Use at most two short items per list (20 words each) and a
justification of at most 50 words so the JSON stays compact."""
        user = (
            f"QUESTION\n{question}\n\nREFERENCE\n{reference}\n\n"
            f"EVIDENCE\n{evidence}\n\nANSWER\n{answer}"
        )
        judgments = []
        for entry in self.config["models"]:
            value = DirectJudgment.model_validate(self._json_call(entry["id"], system, user))
            claims = len(value.supported_claims) + len(value.unsupported_claims)
            judgments.append(
                {
                    "model": entry["id"],
                    **value.model_dump(),
                    "unsupported_atomic_claim_rate": len(value.unsupported_claims) / claims
                    if claims
                    else 0.0,
                }
            )
        scores = [
            0.4 * row["correctness"]
            + 0.25 * row["completeness"]
            + 0.35 * row["evidence_faithfulness"]
            for row in judgments
        ]
        scores.sort()
        unsupported_rates = sorted(row["unsupported_atomic_claim_rate"] for row in judgments)
        return {
            "judges": judgments,
            "median_weighted_score_0_4": scores[len(scores) // 2],
            "score_range": max(scores) - min(scores),
            "disagreement_flag": max(scores) - min(scores) >= 1.0,
            "median_unsupported_atomic_claim_rate": unsupported_rates[len(unsupported_rates) // 2],
        }

    def pairwise(
        self, question_id: str, question: str, answer_left: str, answer_right: str, evidence: str
    ) -> dict:
        system = """Compare two masked biomedical answers against the same supplied evidence.
First state the evidence-based comparison, then a short justification, then choose A, B, or tie.
Do not infer system identity or prefer verbosity. Return JSON keys evidence_comparison,
justification, winner."""
        rng = random.Random(f"20260713:{question_id}")
        initially_swapped = bool(rng.getrandbits(1))
        original = (answer_right, answer_left) if initially_swapped else (answer_left, answer_right)
        judgments = []
        for entry in self.config["models"]:
            mapped: list[str] = []
            raw = []
            for swap in (False, True):
                a, b = (original[1], original[0]) if swap else original
                user = (
                    f"QUESTION\n{question}\n\nEVIDENCE\n{evidence}\n\n"
                    f"ANSWER A\n{a}\n\nANSWER B\n{b}"
                )
                value = PairwiseJudgment.model_validate(self._json_call(entry["id"], system, user))
                winner = value.winner
                if swap and winner != "tie":
                    winner = "A" if winner == "B" else "B"
                raw.append(value.model_dump())
                mapped.append(winner)
            consensus = mapped[0] if mapped[0] == mapped[1] else "tie"
            if initially_swapped and consensus != "tie":
                consensus = "left" if consensus == "B" else "right"
            elif consensus != "tie":
                consensus = "left" if consensus == "A" else "right"
            judgments.append({"model": entry["id"], "winner": consensus, "position_runs": raw})
        counts = {
            value: sum(row["winner"] == value for row in judgments)
            for value in ("left", "right", "tie")
        }
        return {
            "judges": judgments,
            "votes": counts,
            "panel_winner": max(counts, key=lambda name: counts[name]),
        }
