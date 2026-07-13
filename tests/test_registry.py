from medrag_lab.experiments.registry import load_registry, validate_registry


def test_registry_has_registered_counts():
    assert validate_registry(load_registry()) == {"core": 41, "stretch": 13}
