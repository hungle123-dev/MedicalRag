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
    for m in (res.accuracy, res.recall_at_k, res.mrr):
        assert 0.0 <= m <= 1.0
    assert len(res.predictions) == len(QUESTIONS)


def test_e0_retrieves_nothing_so_recall_is_zero():
    res = run_config(_cfg("E0"))
    assert res.recall_at_k == 0.0
    assert res.mrr == 0.0


def test_retrieving_arms_beat_e0_on_recall():
    e0 = run_config(_cfg("E0"))
    for arm in ("E1", "E2"):
        assert run_config(_cfg(arm)).recall_at_k > e0.recall_at_k


def test_accuracy_and_recall_can_diverge():
    # The whole point of the dual metric: E1 finds evidence (high recall) but
    # noisy docs can still yield wrong answers, so accuracy < recall is allowed
    # and expected on this fixture.
    e1 = run_config(_cfg("E1"))
    assert e1.recall_at_k >= e1.accuracy


def test_save_results_persists_config_metrics_and_predictions(tmp_path):
    res = run_config(_cfg("E2"))
    out = tmp_path / "sub" / "results.json"
    save_results([res], str(out), config=_cfg("E2"))
    data = json.loads(out.read_text())
    run = data["runs"][0]
    assert run["arm"] == "E2"
    assert run["n"] == len(QUESTIONS)
    assert {"accuracy", "recall_at_k", "mrr"} <= run.keys()
    assert data["config"]["arm"] == "E2"
    # per-qid audit trail present
    assert len(run["predictions"]) == len(QUESTIONS)
    p0 = run["predictions"][0]
    assert "qid" in p0 and "choice" in p0 and "evidence" in p0


def test_unknown_arm_module_config_still_runs_default():
    # arm_modules omitted -> default module loaded -> E1 works
    res = run_config({"arm": "E1", "dataset": "fixture", "k": 3, "llm": {}})
    assert res.arm_name == "E1"
