import json
import os

from scripts.run import main


def test_run_script_writes_results_and_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # copy shipped configs into cwd-relative path
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for name, arm in (("e0", "E0"), ("e1", "E1"), ("e2", "E2")):
        (cfg_dir / f"{name}.yaml").write_text(
            f"arm: {arm}\ndataset: fixture\nk: 3\nllm:\n  default: A\n"
        )
    rc = main([str(cfg_dir / "e0.yaml"), str(cfg_dir / "e1.yaml"), str(cfg_dir / "e2.yaml")])
    assert rc == 0
    out = capsys.readouterr().out
    for arm in ("E0", "E1", "E2"):
        assert arm in out
    # results written under experiments/<date>_<arm>/
    found = []
    for root, _dirs, files in os.walk(tmp_path / "experiments"):
        found += [os.path.join(root, f) for f in files if f == "results.json"]
    assert len(found) == 3
    data = json.loads(open(found[0]).read())
    assert "runs" in data and "config" in data
