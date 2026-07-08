import pytest

from medgraphrag.config import load_config


def test_loads_valid_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("arm: E1\ndataset: fixture\nk: 5\n")
    cfg = load_config(str(p))
    assert cfg["arm"] == "E1"
    assert cfg["k"] == 5


def test_rejects_config_without_arm(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: fixture\n")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_shipped_configs_are_valid():
    for name in ("e0", "e1", "e2"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["dataset"] == "fixture"
