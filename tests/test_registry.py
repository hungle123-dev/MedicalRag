from medrag_lab.experiments.registry import load_registry, validate_registry


def test_registry_has_registered_counts():
    registry = load_registry()
    assert validate_registry(registry) == {"core": 41, "stretch": 13}
    assert len(registry["resolved_arms"]) == 54
    assert all(arm["config_hash"] for arm in registry["resolved_arms"])
