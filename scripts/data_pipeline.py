"""Download, verify, and profile the pinned public research data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import ijson


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"

SOURCES = {
    "bioasq_corpus": {
        "url": "https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag/resolve/6d6add1a6ec2090991386b5ae7608b71fd637bc4/data/corpus.jsonl",
        "path": RAW / "bioasq" / "corpus.jsonl",
        "size": 126455681,
        "sha256": "1e992bd761413d17b3fbb410a368532227d4bc28d0ef27817dde98ef3ecb2ca0",
    },
    "bioasq_dev": {
        "url": "https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag/resolve/6d6add1a6ec2090991386b5ae7608b71fd637bc4/data/dev.jsonl",
        "path": RAW / "bioasq" / "dev.jsonl",
        "size": 25960520,
        "sha256": "19378f4c5eb4957753bdd7ce67fc570d6d709b6b738b9e0d5ca179e354e0510d",
    },
    "bioasq_eval": {
        "url": "https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag/resolve/6d6add1a6ec2090991386b5ae7608b71fd637bc4/data/eval.jsonl",
        "path": RAW / "bioasq" / "eval.jsonl",
        "size": 4436186,
        "sha256": "61a1521015fc190dcc8c8c0f0d1b3a25ea6a694f915461256557a81c4d4bfbdf",
    },
    "primekg_nodes": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/6180617",
        "path": RAW / "primekg" / "nodes.tab",
        "size": 8893757,
        "source_md5": "7f9ab4109c54049e819ecd14e15a6038",
        "sha256": "18ebcf24887e8529b95c7a7c96c81089cc7c5ba22d53a5c05d56480331961af8",
        "note": "Dataverse returns a derived TSV; source MD5 belongs to original nodes.csv.",
    },
    "primekg_edges": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/6180616",
        "path": RAW / "primekg" / "edges.csv",
        "size": 386582390,
        "md5": "5d4d211a22e88544b78fde2735e797bc",
        "sha256": "57c405049d0adb5ee9070d8b3fbb38f55a053b5c8d45b8b5af1ce23ee5d3ad7c",
    },
    "primekgqa_train": {
        "url": "https://zenodo.org/api/records/13829395/files/train_call_bioLLM.json/content",
        "path": RAW / "primekgqa" / "train_call_bioLLM.json",
        "size": 942393461,
        "md5": "1457be128b9ac4c70fe6139ee01d5b10",
        "sha256": "4f0776ceb2d8890f8b06997744142a150f98255e289d230203307d68f273d627",
    },
    "primekgqa_val": {
        "url": "https://zenodo.org/api/records/13829395/files/val_call_bioLLM.json/content",
        "path": RAW / "primekgqa" / "val_call_bioLLM.json",
        "size": 317780342,
        "md5": "b0404cc5dbe44acce8769a3aeb4f1d76",
        "sha256": "fbb44c55dd2d9feb78788f2276b65b14b922b5963e67bf9fb39ceaeede3aebed",
    },
    "primekgqa_test": {
        "url": "https://zenodo.org/api/records/13829395/files/test_call_bioLLM.json/content",
        "path": RAW / "primekgqa" / "test_call_bioLLM.json",
        "size": 287302216,
        "md5": "43c40df023e7e199774cfb37758a776b",
        "sha256": "603bdc2b5efe5bfc5c0f58925810abca32dbc37cdcb7ffe34a7354d495b455bd",
    },
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_one(name: str) -> dict:
    source = SOURCES[name]
    path: Path = source["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != source["size"]:
        temporary = path.with_suffix(path.suffix + ".part")
        if path.exists() and not temporary.exists():
            os.replace(path, temporary)
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "MedicalRag/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(source["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            resume = offset > 0 and response.status == 206
            mode, copied = ("ab", offset) if resume else ("wb", 0)
            print(f"{name}: {'resuming' if resume else 'downloading'} at {copied} bytes")
            output = temporary.open(mode)
            try:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    copied += len(chunk)
            finally:
                output.close()
        if copied != source["size"]:
            raise ValueError(f"{name}: incomplete download ({copied} != {source['size']}); rerun to resume")
        os.replace(temporary, path)
    if path.stat().st_size != source["size"]:
        raise ValueError(f"{name}: size mismatch ({path.stat().st_size} != {source['size']})")
    md5 = digest(path, "md5")
    if source.get("md5") and md5 != source["md5"]:
        raise ValueError(f"{name}: MD5 mismatch")
    sha256 = digest(path)
    if source.get("sha256") and sha256 != source["sha256"]:
        raise ValueError(f"{name}: SHA-256 mismatch")
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "url": source["url"],
        "bytes": path.stat().st_size,
        "md5": md5,
        "sha256": sha256,
        "source_md5": source.get("source_md5", source.get("md5")),
        "note": source.get("note"),
    }


def describe(numbers: list[int]) -> dict:
    if not numbers:
        return {"count": 0}
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    median_value = (ordered[middle] if len(ordered) % 2
                    else (ordered[middle - 1] + ordered[middle]) / 2)
    return {
        "count": len(numbers),
        "min": ordered[0],
        "median": median_value,
        "mean": round(sum(ordered) / len(ordered), 2),
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "max": ordered[-1],
    }


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def bioasq_eda() -> dict:
    report: dict[str, dict] = {}
    corpus_pmids: set[str] = set()
    for split in ("corpus", "dev", "eval"):
        path = RAW / "bioasq" / f"{split}.jsonl"
        rows = list(read_jsonl(path))
        keys = sorted({key for row in rows for key in row})
        entry = {"rows": len(rows), "fields": keys}
        if split == "corpus":
            ids = [str(row.get("passage_id") or row.get("pmid") or row.get("id") or "") for row in rows]
            corpus_pmids = {value for value in ids if value}
            texts = [str(row.get("passage") or row.get("text") or row.get("abstract") or "") for row in rows]
            titles = [str(row.get("title") or "") for row in rows]
            years = Counter(str(row.get("publication_date") or "")[:4] for row in rows
                            if str(row.get("publication_date") or "")[:4].isdigit())
            entry |= {
                "unique_ids": len(corpus_pmids),
                "duplicate_ids": len(ids) - len(set(ids)),
                "text_chars": describe([len(text) for text in texts]),
                "empty_text": sum(not text.strip() for text in texts),
                "title_chars": describe([len(title) for title in titles]),
                "missing_title": sum(not title.strip() for title in titles),
                "missing_doi": sum(not str(row.get("doi") or "").strip() for row in rows),
                "publication_year_range": [min(years) if years else None, max(years) if years else None],
                "publication_year_top": dict(years.most_common(10)),
            }
        else:
            questions = [str(row.get("question", "")) for row in rows]
            answers = [str(row.get("answer", "")) for row in rows]
            relevant = [list(map(str, row.get("relevant_passage_ids", []))) for row in rows]
            snippets = [snippet for row in rows for snippet in row.get("snippets", [])]
            snippet_texts = [str(snippet.get("text", "")) for snippet in snippets]
            snippet_documents = [snippet.get("document", "").rstrip("/").rsplit("/", 1)[-1]
                                 for snippet in snippets]
            answer_lengths_by_type = {
                kind: describe([len(str(row.get("answer", ""))) for row in rows
                                if str(row.get("type", "missing")) == kind])
                for kind in sorted({str(row.get("type", "missing")) for row in rows})
            }
            all_gold = [pmid for values in relevant for pmid in values]
            entry |= {
                "question_chars": describe([len(value) for value in questions]),
                "answer_chars": describe([len(value) for value in answers]),
                "answer_chars_by_question_type": answer_lengths_by_type,
                "empty_questions": sum(not value.strip() for value in questions),
                "empty_answers": sum(not value.strip() for value in answers),
                "exact_duplicate_questions": len(questions) - len(set(questions)),
                "question_types": dict(Counter(str(row.get("type", "missing")) for row in rows)),
                "gold_passages_per_question": describe([len(value) for value in relevant]),
                "questions_with_any_gold_in_corpus_rate": round(
                    sum(bool(set(values) & corpus_pmids) for values in relevant) / max(len(relevant), 1), 6
                ),
                "gold_passage_id_coverage": round(
                    sum(pmid in corpus_pmids for pmid in all_gold) / max(len(all_gold), 1), 6),
                "snippets_per_question": describe([len(row.get("snippets", [])) for row in rows]),
                "snippet_chars": describe([len(value) for value in snippet_texts]),
                "snippet_sections": dict(Counter(str(snippet.get("beginSection", "missing"))
                                                  for snippet in snippets)),
                "empty_snippets": sum(not value.strip() for value in snippet_texts),
                "duplicate_normalized_snippets": len(snippet_texts) - len({normalize.casefold().strip()
                                                                           for normalize in snippet_texts}),
                "snippet_document_in_corpus_rate": round(
                    sum(pmid in corpus_pmids for pmid in snippet_documents) /
                    max(len(snippet_documents), 1), 6),
            }
        report[split] = entry
    dev_questions = {row.get("question", "").strip().casefold() for row in read_jsonl(RAW / "bioasq" / "dev.jsonl")}
    eval_questions = {row.get("question", "").strip().casefold() for row in read_jsonl(RAW / "bioasq" / "eval.jsonl")}
    report["leakage"] = {"exact_question_overlap_dev_eval": len(dev_questions & eval_questions)}
    return report


def primekg_eda() -> dict:
    node_types, sources, names, indexes = Counter(), Counter(), set(), set()
    missing_names = duplicate_names = 0
    with (RAW / "primekg" / "nodes.tab").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            indexes.add(row["node_index"])
            node_types[row["node_type"]] += 1
            sources[row["node_source"]] += 1
            name = row["node_name"].strip().casefold()
            missing_names += not bool(name)
            duplicate_names += name in names
            names.add(name)
    relations, display_relations, degree = Counter(), Counter(), Counter()
    missing_endpoint = 0
    with (RAW / "primekg" / "edges.csv").open(encoding="utf-8", newline="") as stream:
        edge_count = 0
        for row in csv.DictReader(stream):
            edge_count += 1
            relations[row["relation"]] += 1
            display_relations[row["display_relation"]] += 1
            degree[row["x_index"]] += 1
            degree[row["y_index"]] += 1
            missing_endpoint += row["x_index"] not in indexes or row["y_index"] not in indexes
    return {
        "nodes": len(indexes),
        "edges": edge_count,
        "node_types": dict(node_types),
        "node_sources": dict(sources),
        "relations": dict(relations),
        "display_relations": dict(display_relations),
        "missing_names": missing_names,
        "duplicate_normalized_names": duplicate_names,
        "missing_edge_endpoints": missing_endpoint,
        "degree": describe(list(degree.values())),
        "top_hubs": degree.most_common(20),
    }


IRI = re.compile(r"https://zitniklab\.hms\.harvard\.edu/projects/PrimeKG/(node|vocab)/([^>]+)")


def primekgqa_eda() -> dict:
    report: dict[str, dict] = {}
    all_questions: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = RAW / "primekgqa" / f"{split}_call_bioLLM.json"
        rows = ijson.items(path.open("rb"), "item")
        types, node_counts, answer_sizes, answer_kinds = Counter(), Counter(), [], Counter()
        count = malformed = missing_question = 0
        fields: set[str] = set()
        questions: set[str] = set()
        entity_patterns, relation_patterns = set(), set()
        for row in rows:
            count += 1
            fields.update(row)
            values = row.get("value") or []
            types[str(row.get("type", "missing"))] += 1
            nodes = {part for triple in values for part in (triple[0], triple[2])} if values else set()
            node_counts[len(nodes)] += 1
            answers = row.get("answer_sparql") or []
            answer_sizes.append(len(answers))
            sparql = str(row.get("sparql", ""))
            resources = IRI.findall(sparql)
            answer_kinds["relation" if "?uri  <" not in sparql and ">  ?uri  <" in sparql else "node"] += 1
            malformed += not bool(values and answers and sparql and "SELECT" in sparql.upper())
            question = str(row.get("question") or row.get("generated_question") or row.get("nl_question") or "")
            missing_question += not bool(question.strip())
            if question:
                questions.add(question.strip().casefold())
            entity_patterns.add(tuple(sorted(unquote(value) for kind, value in resources if kind == "node")))
            relation_patterns.add(tuple(sorted(unquote(value) for kind, value in resources if kind == "vocab")))
        all_questions[split] = questions
        report[split] = {
            "rows": count,
            "fields": sorted(fields),
            "types": dict(types),
            "node_counts": dict(sorted(node_counts.items())),
            "answer_types_estimated": dict(answer_kinds),
            "answers_per_question": describe(answer_sizes),
            "malformed_records": malformed,
            "missing_natural_language_question": missing_question,
            "unique_natural_language_questions": len(questions),
            "unique_entity_patterns": len(entity_patterns),
            "unique_relation_patterns": len(relation_patterns),
        }
    report["release_total"] = sum(report[split]["rows"] for split in ("train", "val", "test"))
    report["published_count_discrepancy"] = {"zenodo_conclusion": 83999, "paper_table": 85368}
    report["exact_question_overlap"] = {
        "train_val": len(all_questions["train"] & all_questions["val"]),
        "train_test": len(all_questions["train"] & all_questions["test"]),
        "val_test": len(all_questions["val"] & all_questions["test"]),
    }
    return report


def write_json(name: str, value: object) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def command_download(names: list[str]) -> None:
    selected = list(SOURCES) if names == ["all"] else names
    target = MANIFESTS / "files.json"
    existing = json.loads(target.read_text(encoding="utf-8"))["files"] if target.exists() else []
    merged = {row["name"]: row for row in existing}
    merged.update({row["name"]: row for row in (download_one(name) for name in selected)})
    write_json("files.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(),
                              "files": [merged[name] for name in sorted(merged)]})


def command_eda() -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bioasq": bioasq_eda(),
        "primekg": primekg_eda(),
        "primekgqa": primekgqa_eda(),
    }
    write_json("eda.json", report)
    write_json("counts.json", {
        "bioasq": {key: value["rows"] for key, value in report["bioasq"].items() if isinstance(value, dict) and "rows" in value},
        "primekg": {key: report["primekg"][key] for key in ("nodes", "edges")},
        "primekgqa": {key: report["primekgqa"][key]["rows"] for key in ("train", "val", "test")},
    })
    print(json.dumps(report["primekgqa"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("names", nargs="+", choices=["all", *SOURCES])
    subparsers.add_parser("eda")
    args = parser.parse_args()
    if args.command == "download":
        command_download(args.names)
    else:
        command_eda()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
