import json

from medgraphrag.runner import run_config, save_results, RunResult
from medgraphrag.data.fixture_dataset import QUESTIONS


def _cfg(arm):
    return {"arm": arm, "dataset": "fixture", "k": 3, "llm": {"default": "A"}}


def test_run_config_returns_all_metrics():
    res = run_config(_cfg("E1"))
    assert isinstance(res, RunResult)
    assert res.arm_name == "E1"
    assert res.dataset == "fixture"
    for m in (res.accuracy, res.retrieval_recall, res.mrr):
        assert 0.0 <= m <= 1.0
    assert len(res.predictions) == len(QUESTIONS)


def test_e0_retrieves_nothing_so_recall_is_zero():
    res = run_config(_cfg("E0"))
    assert res.retrieval_recall == 0.0
    assert res.mrr == 0.0


def test_retrieving_arms_beat_e0_on_recall():
    e0 = run_config(_cfg("E0"))
    for arm in ("E1", "E2"):
        assert run_config(_cfg(arm)).retrieval_recall > e0.retrieval_recall


def test_e1_strictly_diverges_recall_above_accuracy():
    # Not just recall >= accuracy (which could hold trivially): assert the gap
    # is REAL on this fixture — E1 finds evidence for more questions than it
    # answers correctly, because distractor docs mislead the reader.
    e1 = run_config(_cfg("E1"))
    assert e1.retrieval_recall > e1.accuracy


def test_q2_is_a_concrete_retrieval_hit_but_wrong_answer():
    # Locks the exact divergence case: for q2 the gold term IS retrieved
    # (recall hit) yet the predicted choice is wrong. This is the evidence that
    # accuracy and grounding are genuinely separate signals, not a tautology.
    res = run_config(_cfg("E1"))
    q2 = next(p for p in res.predictions if p.qid == "q2")
    gold_q2 = next(q for q in QUESTIONS if q.qid == "q2")
    hit = any(gold_q2.gold_terms[0].lower() in e.content.lower() for e in q2.evidence)
    assert hit, "expected q2 evidence to contain the gold term"
    assert q2.choice != gold_q2.answer, "expected q2 answer to be wrong despite the hit"


def test_save_results_persists_config_metrics_and_predictions(tmp_path):
    res = run_config(_cfg("E2"))
    out = tmp_path / "sub" / "results.json"
    save_results([res], str(out), config=_cfg("E2"))
    data = json.loads(out.read_text())
    run = data["runs"][0]
    assert run["arm"] == "E2"
    assert run["n"] == len(QUESTIONS)
    assert {"accuracy", "retrieval_recall", "mrr"} <= run.keys()
    assert data["config"]["arm"] == "E2"
    # per-qid audit trail present
    assert len(run["predictions"]) == len(QUESTIONS)
    p0 = run["predictions"][0]
    assert "qid" in p0 and "choice" in p0 and "evidence" in p0


def test_unknown_arm_module_config_still_runs_default():
    # arm_modules omitted -> default module loaded -> E1 works
    res = run_config({"arm": "E1", "dataset": "fixture", "k": 3, "llm": {}})
    assert res.arm_name == "E1"
