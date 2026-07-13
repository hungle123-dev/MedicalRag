from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import stable_hash
from medrag_lab.evaluation.bioasq import snippet_span_f1
from medrag_lab.evaluation.errors import ERROR_TAXONOMY
from medrag_lab.settings import ROOT, settings


def build_error_audit(
    contexts_path: Path,
    generation_path: Path,
    population: str,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Apply deterministic error labels; unsupported/safety labels remain panel-only."""
    contexts = {str(row["question_id"]): row for row in iter_jsonl(contexts_path)}
    generated = {str(row["question_id"]): row for row in iter_jsonl(generation_path)}
    if set(contexts) != set(generated):
        raise ValueError("Error audit requires identical context and generation question IDs")
    source = settings().medrag_data_dir / (
        "eval.jsonl" if population == "heldout340" else "dev.jsonl"
    )
    gold = {
        str(row["question_id"]): row
        for row in iter_jsonl(source)
        if str(row["question_id"]) in contexts
    }
    if set(gold) != set(contexts):
        raise ValueError("Could not resolve all gold rows for error analysis")
    audited = []
    counts: Counter[str] = Counter()
    for question_id in sorted(contexts):
        context, answer, reference = (
            contexts[question_id],
            generated[question_id],
            gold[question_id],
        )
        codes: list[str] = []
        expected_pmids = set(map(str, reference["relevant_passage_ids"]))
        retrieved_pmids = set(map(str, context.get("retrieved_pmids", [])))
        packed = context.get("packed_evidence", [])
        packed_pmids = {str(item["pmid"]) for item in packed}
        predicted_snippets = [
            {
                "document": f"https://pubmed.ncbi.nlm.nih.gov/{item['pmid']}/",
                "beginSection": item.get("section", "abstract"),
                "offsetInBeginSection": item.get("begin"),
                "offsetInEndSection": item.get("end"),
            }
            for item in packed
        ]
        span_f1 = snippet_span_f1(predicted_snippets, reference["snippets"])["f1"]
        if context.get("failed") or answer.get("failed"):
            codes.append("SYS1")
        if not retrieved_pmids & expected_pmids:
            codes.append("R1")
        elif not packed_pmids & expected_pmids:
            codes.append("CTX1")
        elif span_f1 == 0:
            codes.append("R2")
        answer_value = answer.get("answer") or {}
        if answer_value and answer_value.get("predicted_type") != reference["type"]:
            codes.append("G3")
        if int(answer.get("valid_citation_count", 0)) < int(answer.get("citation_count", 0)):
            codes.append("CIT2")
        rouge_f1 = float(answer.get("rouge_su4", {}).get("f1", 0.0))
        if not answer.get("failed") and rouge_f1 < 0.10:
            codes.append("G2")
        counts.update(codes)
        audited.append(
            {
                "question_id": question_id,
                "codes": codes,
                "snippet_span_f1": span_f1,
                "rouge_su4_f1": rouge_f1,
            }
        )
    result = {
        "status": "deterministic_exploratory_error_audit",
        "population": population,
        "questions": len(audited),
        "thresholds": {"G2_rouge_su4_f1_below": 0.10},
        "counts": dict(sorted(counts.items())),
        "taxonomy": ERROR_TAXONOMY,
        "unassigned_without_llm_panel": ["G1", "CIT1", "S1", "CTX2"],
        "rows": audited,
    }
    result["audit_hash"] = stable_hash(result)
    output = destination or ROOT / "reports" / "error_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
