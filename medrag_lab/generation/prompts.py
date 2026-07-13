from __future__ import annotations

import hashlib
import json

SYSTEM_PROMPT = """You are a biomedical research question-answering system.
Use only the supplied evidence. Do not invent citations, diagnoses, treatment instructions,
or patient-specific advice. If evidence is insufficient, abstain. Return one JSON object only.

Output schema:
{
  "predicted_type": "yesno|factoid|list|summary",
  "exact_answer": "yes/no" | ["ranked factoid/list entries"] | null,
  "ideal_answer": "concise biomedical answer, at most 200 words, without citation markup",
  "citation_pmids": ["digits from supplied PMID labels only"],
  "abstained": false,
  "evidence_support_score": 0.0
}

For summary questions exact_answer must be null. A support score is an evidence diagnostic,
not a calibrated probability of correctness."""

CLOSED_BOOK_SYSTEM_PROMPT = """You are a biomedical research question-answering system.
Answer from pretrained knowledge without retrieved evidence. Do not invent citations and always
return citation_pmids=[]. This is a closed-book experimental baseline, not medical advice. Return
the same JSON schema as the evidence-grounded system. Set evidence_support_score=null."""


def answer_prompt(question: str, context: str) -> str:
    return f"QUESTION\n{question}\n\nEVIDENCE\n{context or '[NO EVIDENCE]'}"


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(
        json.dumps({"system": system, "user": user}, sort_keys=True).encode()
    ).hexdigest()
