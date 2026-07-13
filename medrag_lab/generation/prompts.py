from __future__ import annotations

import hashlib
import json

GENERIC_STRUCTURED_SYSTEM_PROMPT = """You are a biomedical research question-answering system.
Use only the supplied evidence and return one JSON object matching this schema:
{
  "predicted_type": "yesno|factoid|list|summary",
  "exact_answer": "yes/no" | ["ranked factoid/list entries"] | null,
  "ideal_answer": "concise biomedical answer, at most 200 words",
  "citation_pmids": ["PMID strings"],
  "abstained": false,
  "evidence_support_score": 0.0
}
Do not provide patient-specific medical advice."""

CITATION_SYSTEM_PROMPT = (
    GENERIC_STRUCTURED_SYSTEM_PROMPT
    + """

Every citation_pmids entry must be copied from a [PMID:...] label in the supplied evidence.
Do not invent citations. If the evidence is insufficient, set abstained=true and still provide
the non-empty ideal_answer "Insufficient evidence in the provided context."."""
)

SYSTEM_PROMPT = (
    CITATION_SYSTEM_PROMPT
    + """

Infer the BioASQ question type before answering. For summary questions exact_answer must be null.
For yes/no questions exact_answer must be exactly yes or no. For factoid/list questions return a
ranked JSON list. evidence_support_score is a diagnostic, not a calibrated probability.
"""
)

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
