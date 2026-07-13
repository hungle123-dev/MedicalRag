from __future__ import annotations

import argparse
import json

from medrag_lab.data.audit import audit_bundle, write_audit
from medrag_lab.data.splits import freeze_splits, verify_splits
from medrag_lab.experiments.registry import load_registry, validate_registry
from medrag_lab.experiments.runner import build_bm25, run_bm25


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
    return root


def main() -> None:
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
    elif args.command == "experiment" and args.action == "validate":
        registry = load_registry()
        print(
            json.dumps(validate_registry(registry) | {"registry_hash": registry["registry_hash"]})
        )
    elif args.command == "experiment" and args.action == "bm25":
        print(json.dumps(run_bm25(args.recipe, args.population, args.limit), indent=2))
