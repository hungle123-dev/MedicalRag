from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from medrag_lab.data.audit import audit_bundle, write_audit
from medrag_lab.data.splits import freeze_splits, verify_splits
from medrag_lab.evaluation.report import build_report
from medrag_lab.experiments.final import freeze_finalists, verify_final_freeze
from medrag_lab.experiments.registry import load_registry, validate_registry
from medrag_lab.experiments.runner import (
    build_bm25,
    compare_prediction_files,
    run_bm25,
    run_dense_retrieval,
    run_generation,
    run_oracle,
)
from medrag_lab.indexing.medcpt import build_index as build_medcpt


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="medrag")
    commands = root.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="action", required=True)
    data_commands.add_parser("audit")
    freeze = data_commands.add_parser("freeze")
    freeze.add_argument("--seed", type=int, default=20260713)
    data_commands.add_parser("verify")

    index = commands.add_parser("index")
    index_commands = index.add_subparsers(dest="action", required=True)
    build = index_commands.add_parser("build-bm25")
    build.add_argument(
        "--recipe",
        choices=("title", "abstract", "title_abstract", "boosted_title_abstract_mesh"),
        default="title_abstract",
    )
    build.add_argument("--force", action="store_true")
    dense = index_commands.add_parser("build-medcpt")
    dense.add_argument("--batch-size", type=int, default=16)
    dense.add_argument("--force", action="store_true")

    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(dest="action", required=True)
    experiment_commands.add_parser("validate")
    bm25 = experiment_commands.add_parser("bm25")
    bm25.add_argument(
        "--recipe",
        choices=("title", "abstract", "title_abstract", "boosted_title_abstract_mesh"),
        default="title_abstract",
    )
    bm25.add_argument("--population", default="smoke40")
    bm25.add_argument("--limit", type=int)
    retrieval = experiment_commands.add_parser("retrieval")
    retrieval.add_argument("--method", choices=("medcpt", "rrf", "rrf_rerank"), required=True)
    retrieval.add_argument("--population", default="smoke40")
    retrieval.add_argument("--limit", type=int)
    generation = experiment_commands.add_parser("generation")
    generation.add_argument("--pipeline", required=True)
    generation.add_argument("--population", default="smoke40")
    generation.add_argument("--limit", type=int)
    final_freeze = experiment_commands.add_parser("freeze-final")
    final_freeze.add_argument("--verify", action="store_true")
    oracle = experiment_commands.add_parser("oracle")
    oracle.add_argument("--population", default="validation200")
    oracle.add_argument("--limit", type=int)
    comparison = experiment_commands.add_parser("compare")
    comparison.add_argument("--left", required=True)
    comparison.add_argument("--right", required=True)
    comparison.add_argument("--metric", default="metrics.ap")
    commands.add_parser("report")
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args()
    if args.command == "data" and args.action == "audit":
        report = audit_bundle()
        manifest, html = write_audit(report)
        print(json.dumps({"status": "ok", "manifest": str(manifest), "eda": str(html)}))
    elif args.command == "data" and args.action == "freeze":
        result = freeze_splits(args.seed)
        print(json.dumps({"status": "ok", "freeze_hash": result["freeze_hash"]}))
    elif args.command == "data" and args.action == "verify":
        print(json.dumps(verify_splits()))
    elif args.command == "index" and args.action == "build-bm25":
        print(build_bm25(args.recipe, args.force))
    elif args.command == "index" and args.action == "build-medcpt":
        result = build_medcpt(batch_size=args.batch_size, force=args.force)
        print(json.dumps({key: str(value) for key, value in vars(result).items()}))
    elif args.command == "experiment" and args.action == "validate":
        registry = load_registry()
        print(
            json.dumps(validate_registry(registry) | {"registry_hash": registry["registry_hash"]})
        )
    elif args.command == "experiment" and args.action == "bm25":
        print(json.dumps(run_bm25(args.recipe, args.population, args.limit), indent=2))
    elif args.command == "experiment" and args.action == "retrieval":
        print(json.dumps(run_dense_retrieval(args.method, args.population, args.limit), indent=2))
    elif args.command == "experiment" and args.action == "generation":
        print(json.dumps(run_generation(args.pipeline, args.population, args.limit), indent=2))
    elif args.command == "experiment" and args.action == "freeze-final":
        result = verify_final_freeze() if args.verify else freeze_finalists()
        print(json.dumps(result, indent=2))
    elif args.command == "experiment" and args.action == "oracle":
        print(json.dumps(run_oracle(args.population, args.limit), indent=2))
    elif args.command == "experiment" and args.action == "compare":
        print(
            json.dumps(
                compare_prediction_files(Path(args.left), Path(args.right), metric=args.metric),
                indent=2,
            )
        )
    elif args.command == "report":
        print(build_report())
