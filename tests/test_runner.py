import json

from medgraphrag.runner import run_config, save_results, RunResult
from tests.fixtures.mini_mirage import QUESTIONS


def _cfg(arm):
    return {"arm": arm, "dataset": "fixture", "k": 3, "llm": {"default": "A"}}


def test_run_config_returns_result():
    res = run_config(_cfg("E1"))
    assert isinstance(res, RunResult)
    assert res.arm_name == "E1"
    assert res.dataset == "fixture"
    assert 0.0 <= res.accuracy <= 1.0
    assert len(res.predictions) == len(QUESTIONS)


def test_e1_beats_e0_on_fixture():
    assert run_config(_cfg("E1")).accuracy > run_config(_cfg("E0")).accuracy


def test_e2_beats_e0_on_fixture():
    assert run_config(_cfg("E2")).accuracy > run_config(_cfg("E0")).accuracy


def test_save_results_writes_json_with_config(tmp_path):
    res = run_config(_cfg("E1"))
    out = tmp_path / "sub" / "results.json"
    save_results([res], str(out), config=_cfg("E1"))
    data = json.loads(out.read_text())
    assert data["runs"][0]["arm"] == "E1"
    assert data["runs"][0]["n"] == len(QUESTIONS)
    assert data["config"]["arm"] == "E1"
