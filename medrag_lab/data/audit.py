from __future__ import annotations

import html
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medrag_lab.data.loaders import iter_jsonl
from medrag_lab.data.manifests import (
    DATASET_URL,
    LICENSE,
    PINNED_REVISION,
    atomic_json,
    verify_file,
)
from medrag_lab.data.splits import normalize_question
from medrag_lab.settings import ROOT, settings

WORD = re.compile(r"[\w]+(?:[-_/][\w]+)*", re.UNICODE)


def words(text: str) -> list[str]:
    return WORD.findall(text)


def describe(values: Iterable[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def pmid_from_value(value: object) -> str:
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def audit_bundle() -> dict[str, Any]:
    data_dir = settings().medrag_data_dir
    paths = {name: data_dir / f"{name}.jsonl" for name in ("corpus", "dev", "eval")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    corpus_sections: dict[str, tuple[str, str]] = {}
    corpus_ids: list[str] = []
    corpus_lengths: list[int] = []
    corpus_norms: Counter[str] = Counter()
    years: list[int] = []
    missing = Counter()
    corpus_rows = 0
    for row in iter_jsonl(paths["corpus"]):
        corpus_rows += 1
        pmid = str(row.get("id", ""))
        title, abstract = str(row.get("title", "")), str(row.get("text", ""))
        corpus_ids.append(pmid)
        corpus_sections[pmid] = (title, abstract)
        corpus_lengths.append(len(words(f"{title} {abstract}")))
        corpus_norms[normalize_question(f"{title} {abstract}")] += 1
        for field in ("id", "title", "text", "url", "publication_date"):
            missing[field] += not bool(row.get(field))
        missing["mesh_terms"] += not bool(row.get("mesh_terms"))
        year = str(row.get("publication_date", ""))[:4]
        if year.isdigit():
            years.append(int(year))

    corpus_id_set = set(corpus_ids)
    qa_reports: dict[str, Any] = {}
    normalized_by_split: dict[str, set[str]] = {}
    all_gold_pmids: set[str] = set()
    file_rows = {"corpus": corpus_rows}

    for split in ("dev", "eval"):
        rows = list(iter_jsonl(paths[split]))
        file_rows[split] = len(rows)
        normalized = [normalize_question(str(row.get("question", ""))) for row in rows]
        normalized_by_split[split] = set(normalized)
        gold_refs = [str(pmid) for row in rows for pmid in row.get("relevant_passage_ids", [])]
        all_gold_pmids.update(gold_refs)
        snippets = [snippet for row in rows for snippet in row.get("snippets", [])]
        reachable_documents = contained_text = exact_offsets = unknown_section = 0
        for snippet in snippets:
            sections = corpus_sections.get(pmid_from_value(snippet.get("document")))
            if not sections:
                continue
            reachable_documents += 1
            section_name = str(snippet.get("beginSection", "")).casefold()
            section = (
                sections[0]
                if section_name == "title"
                else sections[1]
                if section_name == "abstract"
                else ""
            )
            if not section:
                unknown_section += 1
                continue
            text = str(snippet.get("text", ""))
            contained_text += bool(text and text in section)
            begin, end = snippet.get("offsetInBeginSection"), snippet.get("offsetInEndSection")
            if (
                isinstance(begin, int)
                and isinstance(end, int)
                and 0 <= begin <= end <= len(section)
            ):
                exact_offsets += section[begin:end] == text

        counts = Counter(normalized)
        reachable_refs = sum(pmid in corpus_id_set for pmid in gold_refs)
        qa_reports[split] = {
            "rows": len(rows),
            "fields": sorted({field for row in rows for field in row}),
            "question_types": dict(sorted(Counter(str(row.get("type")) for row in rows).items())),
            "question_words": describe(len(words(str(row.get("question", "")))) for row in rows),
            "ideal_answer_words": describe(len(words(str(row.get("answer", "")))) for row in rows),
            "gold_documents_per_question": describe(
                len(row.get("relevant_passage_ids", [])) for row in rows
            ),
            "snippets_per_question": describe(len(row.get("snippets", [])) for row in rows),
            "normalized_duplicate_groups": sum(count > 1 for count in counts.values()),
            "normalized_duplicate_rows": sum(count for count in counts.values() if count > 1),
            "exact_answer_rows": sum(row.get("exact_answer") is not None for row in rows),
            "gold_document_refs": len(gold_refs),
            "gold_document_refs_reachable": reachable_refs,
            "gold_document_coverage": round(reachable_refs / len(gold_refs), 8),
            "questions_with_no_reachable_gold": sum(
                not (set(map(str, row.get("relevant_passage_ids", []))) & corpus_id_set)
                for row in rows
            ),
            "snippet_refs": len(snippets),
            "snippet_documents_reachable": reachable_documents,
            "snippet_text_contained": contained_text,
            "snippet_offset_exact": exact_offsets,
            "snippet_missing_or_unknown_section": unknown_section,
        }

    files = {
        paths[name].name: verify_file(paths[name], file_rows[name])
        for name in ("corpus", "dev", "eval")
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "observed_from_pinned_local_data",
        "dataset": {
            "name": "mattmorgis/bioasq-12b-rag",
            "revision": PINNED_REVISION,
            "url": DATASET_URL,
            "license": LICENSE,
            "scope": "closed-world positive-only gold-conditioned candidate pool",
        },
        "files": files,
        "corpus": {
            "rows": corpus_rows,
            "unique_pmids": len(corpus_id_set),
            "duplicate_pmids": len(corpus_ids) - len(corpus_id_set),
            "document_words": describe(corpus_lengths),
            "missing": dict(missing),
            "publication_year": describe(years),
            "normalized_duplicate_groups": sum(count > 1 for count in corpus_norms.values()),
            "normalized_duplicate_rows": sum(count for count in corpus_norms.values() if count > 1),
        },
        "qa": qa_reports,
        "integrity": {
            "normalized_question_overlap_dev_eval": len(
                normalized_by_split["dev"] & normalized_by_split["eval"]
            ),
            "corpus_pmids_in_dev_eval_gold_union": len(corpus_id_set & all_gold_pmids),
            "corpus_pmids_outside_dev_eval_gold_union": len(corpus_id_set - all_gold_pmids),
        },
        "label_gate": {
            "status": "blocked_missing_official_exact_labels",
            "exact_metrics_enabled": False,
            "official_training_count": 5_046,
            "local_dev_count": 5_049,
            "reason": (
                "The local bundle contains ideal prose but no official exact_answer. "
                "Authorized official labels must be joined by question_id and are never inferred."
            ),
        },
    }
    return report


def write_audit(report: dict[str, Any]) -> tuple[Path, Path]:
    manifest = ROOT / "data" / "manifests" / "bioasq.json"
    report_json = ROOT / "reports" / "eda.json"
    report_html = ROOT / "reports" / "EDA.html"
    atomic_json(manifest, report)
    atomic_json(report_json, report)
    cards = {
        "Corpus": report["corpus"]["rows"],
        "Development": report["qa"]["dev"]["rows"],
        "Held-out": report["qa"]["eval"]["rows"],
        "Corpus outside gold union": report["integrity"][
            "corpus_pmids_outside_dev_eval_gold_union"
        ],
    }
    rendered_cards = "".join(
        f"<article><strong>{html.escape(label)}</strong><span>{value:,}</span></article>"
        for label, value in cards.items()
    )
    report_html.parent.mkdir(parents=True, exist_ok=True)
    machine_report = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    created_at = html.escape(report["created_at"])
    label_reason = html.escape(report["label_gate"]["reason"])
    report_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>MedRAG-Lab EDA</title>
  <style>
    body {{ font: 16px/1.55 system-ui; max-width: 1100px; margin: auto; padding: 32px;
      background: #f5f7f3; color: #172018; }}
    h1 {{ font-size: clamp(2rem, 6vw, 4rem); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px; }}
    article, pre, .warn {{ background: white; border: 1px solid #ccd4ca;
      border-radius: 12px; padding: 18px; }}
    article span {{ display: block; font-size: 2rem; }}
    .warn {{ border-left: 5px solid #ad4e48; }}
    pre {{ overflow: auto; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>BioASQ data audit</h1>
  <p>Generated from pinned local files at {created_at}.</p>
  <div class="cards">{rendered_cards}</div>
  <p class="warn"><strong>Scope:</strong> this is a positive-only gold-conditioned pool,
    not search over all PubMed.</p>
  <h2>Exact-answer gate</h2><p>{label_reason}</p>
  <h2>Machine-readable evidence</h2><pre>{machine_report}</pre>
</body>
</html>""",
        encoding="utf-8",
    )
    return manifest, report_html
