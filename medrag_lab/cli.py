from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from medrag_lab.data.audit import audit_bundle, write_audit
from medrag_lab.data.splits import freeze_splits, verify_splits
from medrag_lab.evaluation.error_audit import build_error_audit
from medrag_lab.evaluation.panel_runner import run_panel_direct, run_panel_pairwise
from medrag_lab.evaluation.report import build_report
from medrag_lab.experiments.analysis import analyze_two_by_two_interaction
from medrag_lab.experiments.evidence import run_evidence_retrieval
from medrag_lab.experiments.final import apply_final_holm, freeze_finalists, verify_final_freeze
from medrag_lab.experiments.generation import prepare_contexts, run_context_generation
from medrag_lab.experiments.registry import load_registry, validate_registry
from medrag_lab.experiments.runner import (
    build_bm25,
    compare_prediction_files,
    evaluate_superiority_gate,
    run_bm25,
    run_dense_retrieval,
    run_generation,
    run_judge_sanity,
    run_oracle,
    run_query_retrieval,
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
    retrieval.add_argument(
        "--bm25-recipe",
        choices=("title", "abstract", "title_abstract", "boosted_title_abstract_mesh"),
        default="title_abstract",
    )
    generation = experiment_commands.add_parser("generation")
    generation.add_argument("--pipeline", required=True)
    generation.add_argument("--population", default="smoke40")
    generation.add_argument("--limit", type=int)
    final_freeze = experiment_commands.add_parser("freeze-final")
    final_freeze.add_argument("--verify", action="store_true")
    oracle = experiment_commands.add_parser("oracle")
    oracle.add_argument("--population", default="validation200")
    oracle.add_argument("--limit", type=int)
    oracle.add_argument("--pipeline", default="bm25_deepseek")
    comparison = experiment_commands.add_parser("compare")
    comparison.add_argument("--left", required=True)
    comparison.add_argument("--right", required=True)
    comparison.add_argument("--metric", default="metrics.ap")
    comparison.add_argument("--require-equal-evidence", action="store_true")
    experiment_commands.add_parser("judge-sanity")
    gate = experiment_commands.add_parser("gate")
    gate.add_argument("--id", required=True)
    gate.add_argument("--comparison", required=True)
    gate.add_argument("--left-efficiency", required=True)
    gate.add_argument("--right-efficiency", required=True)
    query = experiment_commands.add_parser("query")
    query.add_argument("--strategy", choices=("original", "mesh", "hyde"), required=True)
    query.add_argument("--population", default="query800")
    query.add_argument("--limit", type=int)
    query.add_argument(
        "--bm25-recipe",
        choices=("title", "abstract", "title_abstract", "boosted_title_abstract_mesh"),
        default="boosted_title_abstract_mesh",
    )
    query.add_argument("--retriever", choices=("rrf", "rrf_rerank"), default="rrf")
    query.add_argument("--workers", type=int, default=4)
    evidence = experiment_commands.add_parser("evidence")
    evidence.add_argument(
        "--arm",
        choices=(
            "full_document_fields",
            "fixed256_bm25",
            "sentence3_bm25",
            "sentence3_cross_encoder",
        ),
        required=True,
    )
    evidence.add_argument("--retrieval-predictions", type=Path, required=True)
    evidence.add_argument("--population", default="selection4849")
    evidence.add_argument("--limit", type=int)
    context = experiment_commands.add_parser("prepare-contexts")
    context.add_argument("--family", required=True)
    context.add_argument("--arm", required=True)
    context.add_argument("--pipeline", required=True)
    context.add_argument("--population", default="generation160")
    context.add_argument("--context-budget", type=int, default=1200)
    context.add_argument(
        "--context-order",
        choices=("relevance_descending", "strongest_middle"),
        default="relevance_descending",
    )
    context.add_argument("--diversity", choices=("none", "one_per_pmid"), default="none")
    context.add_argument(
        "--evidence-strategy",
        choices=("full_abstract", "fixed256", "sentence3"),
        default="sentence3",
    )
    context.add_argument("--limit", type=int)
    context_generation = experiment_commands.add_parser("generate-contexts")
    context_generation.add_argument("--family", required=True)
    context_generation.add_argument("--arm", required=True)
    context_generation.add_argument("--contexts", type=Path, required=True)
    context_generation.add_argument("--population", default="generation160")
    context_generation.add_argument("--model", required=True)
    context_generation.add_argument(
        "--prompt-style",
        choices=(
            "generic_structured",
            "citation_constraint",
            "predicted_type_schema",
            "gold_type_oracle",
        ),
        default="predicted_type_schema",
    )
    context_generation.add_argument("--workers", type=int, default=4)
    context_generation.add_argument("--limit", type=int)
    interaction = experiment_commands.add_parser("interaction")
    interaction.add_argument("--id", required=True)
    interaction.add_argument("--a0b0", type=Path, required=True)
    interaction.add_argument("--a0b1", type=Path, required=True)
    interaction.add_argument("--a1b0", type=Path, required=True)
    interaction.add_argument("--a1b1", type=Path, required=True)
    interaction.add_argument("--metric", required=True)
    final_holm = experiment_commands.add_parser("final-holm")
    final_holm.add_argument("--comparison", type=Path, action="append", required=True)
    error_audit = experiment_commands.add_parser("error-audit")
    error_audit.add_argument("--contexts", type=Path, required=True)
    error_audit.add_argument("--generation", type=Path, required=True)
    error_audit.add_argument("--population", required=True)
    panel_direct = experiment_commands.add_parser("panel-direct")
    panel_direct.add_argument("--generation", type=Path, required=True)
    panel_direct.add_argument("--contexts", type=Path, required=True)
    panel_direct.add_argument("--population", required=True)
    panel_direct.add_argument("--limit", type=int)
    panel_direct.add_argument("--workers", type=int, default=2)
    panel_pairwise = experiment_commands.add_parser("panel-pairwise")
    panel_pairwise.add_argument("--left", type=Path, required=True)
    panel_pairwise.add_argument("--right", type=Path, required=True)
    panel_pairwise.add_argument("--contexts", type=Path, required=True)
    panel_pairwise.add_argument("--population", default="judge160")
    panel_pairwise.add_argument("--limit", type=int)
    panel_pairwise.add_argument("--workers", type=int, default=2)
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
        print(
            json.dumps(
                run_dense_retrieval(args.method, args.population, args.limit, args.bm25_recipe),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "generation":
        print(json.dumps(run_generation(args.pipeline, args.population, args.limit), indent=2))
    elif args.command == "experiment" and args.action == "freeze-final":
        result = verify_final_freeze() if args.verify else freeze_finalists()
        print(json.dumps(result, indent=2))
    elif args.command == "experiment" and args.action == "oracle":
        print(json.dumps(run_oracle(args.population, args.limit, args.pipeline), indent=2))
    elif args.command == "experiment" and args.action == "compare":
        print(
            json.dumps(
                compare_prediction_files(
                    Path(args.left),
                    Path(args.right),
                    metric=args.metric,
                    require_equal_evidence=args.require_equal_evidence,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "judge-sanity":
        print(json.dumps(run_judge_sanity(), indent=2))
    elif args.command == "experiment" and args.action == "gate":
        print(
            json.dumps(
                evaluate_superiority_gate(
                    Path(args.comparison),
                    Path(args.left_efficiency),
                    Path(args.right_efficiency),
                    args.id,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "query":
        print(
            json.dumps(
                run_query_retrieval(
                    args.strategy,
                    args.population,
                    args.limit,
                    args.bm25_recipe,
                    args.retriever,
                    args.workers,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "evidence":
        print(
            json.dumps(
                run_evidence_retrieval(
                    args.arm,
                    args.retrieval_predictions,
                    args.population,
                    args.limit,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "prepare-contexts":
        print(
            json.dumps(
                prepare_contexts(
                    args.family,
                    args.arm,
                    args.pipeline,
                    args.population,
                    context_token_budget=args.context_budget,
                    context_order=args.context_order,
                    diversity=args.diversity,
                    evidence_strategy=args.evidence_strategy,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "generate-contexts":
        print(
            json.dumps(
                run_context_generation(
                    args.family,
                    args.arm,
                    args.contexts,
                    args.population,
                    args.model,
                    args.prompt_style,
                    workers=args.workers,
                    limit=args.limit,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "interaction":
        print(
            json.dumps(
                analyze_two_by_two_interaction(
                    args.a0b0,
                    args.a0b1,
                    args.a1b0,
                    args.a1b1,
                    args.metric,
                    args.id,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "final-holm":
        print(json.dumps(apply_final_holm(args.comparison), indent=2))
    elif args.command == "experiment" and args.action == "error-audit":
        print(
            json.dumps(build_error_audit(args.contexts, args.generation, args.population), indent=2)
        )
    elif args.command == "experiment" and args.action == "panel-direct":
        print(
            json.dumps(
                run_panel_direct(
                    args.generation,
                    args.contexts,
                    args.population,
                    limit=args.limit,
                    workers=args.workers,
                ),
                indent=2,
            )
        )
    elif args.command == "experiment" and args.action == "panel-pairwise":
        print(
            json.dumps(
                run_panel_pairwise(
                    args.left,
                    args.right,
                    args.contexts,
                    args.population,
                    limit=args.limit,
                    workers=args.workers,
                ),
                indent=2,
            )
        )
    elif args.command == "report":
        print(build_report())
