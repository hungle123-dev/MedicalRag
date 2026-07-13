import pytest

from medrag_lab.evaluation.bioasq import exact_answer_score, rouge_su4, snippet_span_f1
from medrag_lab.evaluation.retrieval import retrieval_metrics
from medrag_lab.evaluation.statistics import (
    holm_adjust,
    paired_effect_size,
    paired_group_bootstrap,
    paired_permutation_p,
)
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
