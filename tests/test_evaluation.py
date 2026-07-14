import json

import pytest

from medrag_lab.evaluation.bioasq import exact_answer_score, rouge_su4, snippet_span_f1
from medrag_lab.evaluation.error_audit import retrieval_error_codes
from medrag_lab.evaluation.llm_panel import aggregate_panel_winner
from medrag_lab.evaluation.retrieval import retrieval_metrics
from medrag_lab.evaluation.statistics import (
    holm_adjust,
    krippendorff_alpha_ordinal,
    nearest_rank_percentile,
    paired_effect_size,
    paired_group_bootstrap,
    paired_mde_80,
    paired_permutation_p,
)
from medrag_lab.experiments import analysis, runner
from medrag_lab.experiments.gates import noninferiority_gate, superiority_gate


def test_bioasq_ap_denominator_for_fewer_and_more_than_ten_gold():
    fewer = retrieval_metrics(["a", "noise", "b"], {"a", "b"}, 10)
    assert fewer["ap"] == pytest.approx((1.0 + 2 / 3) / 2)
    more_gold = {str(value) for value in range(12)}
    assert retrieval_metrics([str(value) for value in range(10)], more_gold, 10)["ap"] == 1.0


def test_duplicate_pmids_and_missing_predictions_are_not_overcounted():
    assert retrieval_metrics(["a", "a", "b"], {"a", "b"}, 10)["ap"] == 1.0
    assert retrieval_metrics([], {"a"}, 10) == {
        "ap": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "hit": 0.0,
    }


def test_exact_metrics_require_official_gold():
    with pytest.raises(ValueError, match="Official exact gold"):
        exact_answer_score("factoid", ["x"], None)
    assert exact_answer_score("yesno", "Yes", "yes")["accuracy"] == 1.0


def test_rouge_su4_and_span_metric():
    assert rouge_su4("alpha beta gamma", "alpha beta gamma")["f1"] == 1.0
    snippet = {
        "document": "https://pubmed.ncbi.nlm.nih.gov/123/",
        "beginSection": "abstract",
        "offsetInBeginSection": 2,
        "offsetInEndSection": 8,
    }
    assert snippet_span_f1([snippet], [snippet])["f1"] == 1.0


def test_paired_group_bootstrap_and_holm():
    result = paired_group_bootstrap([0, 0, 0], [1, 1, 1], ["a", "b", "b"], resamples=100)
    assert result["ci95_low"] == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_effect_size_and_gates():
    assert paired_effect_size([0.1, 0.2, 0.3], [0.2, 0.31, 0.39]) > 0
    assert superiority_gate(0.02, 0.005, 0, 1.2)["passed"] is True
    assert noninferiority_gate(-0.005, -0.009)["passed"] is True
    assert paired_permutation_p([0] * 8, [1] * 8, resamples=1_000) < 0.02
    assert nearest_rank_percentile([10, 20], 0.95) == 20
    assert paired_mde_80([0, 0, 0], [0.1, 0.2, 0.3], ["a", "b", "c"]) > 0
    assert krippendorff_alpha_ordinal([[0, 0, 0], [3, 3, 3]]) == 1.0


def test_superiority_gate_rejects_batched_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "reports" / "gates").mkdir(parents=True)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_hash": "x",
                "bootstrap": {"mean_delta_right_minus_left": 0.02, "ci95_low": 0.01},
            }
        )
    )
    summaries = []
    for name in ("left", "right"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "config": {"latency_mode": "batched_throughput_amortized"},
                    "metrics": {"failure_rate": 0.0, "latency_ms_p95": 1.0},
                }
            )
        )
        summaries.append(path)
    with pytest.raises(ValueError, match="dedicated serial"):
        runner.evaluate_superiority_gate(comparison, *summaries, "test")


def test_superiority_gate_rejects_missing_latency_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    (tmp_path / "reports" / "gates").mkdir(parents=True)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_hash": "x",
                "bootstrap": {"mean_delta_right_minus_left": 0.02, "ci95_low": 0.01},
            }
        )
    )
    summaries = []
    for name in ("left", "right"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"config": {}, "metrics": {"failure_rate": 0.0, "latency_ms_p95": 1.0}})
        )
        summaries.append(path)
    with pytest.raises(ValueError, match="dedicated serial"):
        runner.evaluate_superiority_gate(comparison, *summaries, "test")


def test_panel_vote_ties_remain_ties():
    assert aggregate_panel_winner({"left": 1, "right": 1, "tie": 1}) == "tie"
    assert aggregate_panel_winner({"left": 1, "right": 2, "tie": 0}) == "right"


def test_retrieval_error_attribution_requires_provenance():
    assert retrieval_error_codes(
        {"gold"}, set(), set(), 0.0, provenance_available=False
    ) == []
    assert retrieval_error_codes(
        {"gold"}, set(), set(), 0.0, provenance_available=True
    ) == ["R1"]


def test_evidence_gate_checks_effect_failures_and_population(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_hash": "c",
                "rows": 10,
                "bootstrap": {"mean_delta_right_minus_left": 0.02, "ci95_low": 0.01},
            }
        )
    )
    summaries = []
    for name in ("baseline", "candidate"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"metrics": {"questions": 10, "failure_rate": 0.0}}))
        summaries.append(path)
    result = analysis.evaluate_evidence_gate(*summaries, comparison, "E04_TEST")
    assert result["passed"] is True


def test_diversity_gate_requires_recall_noninferiority_and_reliability(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps({"metrics": {"failure_rate": 0.0, "citation_validity": 1.0}})
    )
    candidate.write_text(
        json.dumps({"metrics": {"failure_rate": 0.0, "citation_validity": 1.0}})
    )
    answer = tmp_path / "answer.json"
    recall = tmp_path / "recall.json"
    answer.write_text(
        json.dumps(
            {
                "comparison_hash": "a",
                "rows": 10,
                "bootstrap": {"mean_delta_right_minus_left": -0.005, "ci95_low": -0.015},
            }
        )
    )
    recall.write_text(
        json.dumps(
            {
                "comparison_hash": "r",
                "rows": 10,
                "bootstrap": {"mean_delta_right_minus_left": 0.1, "ci95_low": 0.04},
            }
        )
    )
    result = analysis.evaluate_diversity_gate(
        baseline, candidate, answer, recall, "E07_TEST"
    )
    assert result["passed"] is True


def test_generator_screening_gate_rejects_unreliable_route(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "metrics": {
                    "questions": 40,
                    "failure_rate": 0.15,
                    "retry_rate": 0.05,
                    "citation_validity": 1.0,
                    "latency_ms_p95": 5000,
                    "rouge_su4_f1": 0.12,
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            }
        )
    )
    result = analysis.evaluate_generator_screening_gate(summary, "E08_TEST")
    assert result["passed"] is False
    assert result["checks"]["failure_rate"] is False
