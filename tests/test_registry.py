from copy import deepcopy

import pytest

from medrag_lab.cli import parser
from medrag_lab.experiments.registry import load_registry, validate_registry


def test_registry_has_registered_counts():
    registry = load_registry()
    assert validate_registry(registry) == {"core": 41, "stretch": 13}
    assert len(registry["resolved_arms"]) == 54
    assert all(arm["config_hash"] for arm in registry["resolved_arms"])


def test_registry_rejects_unknown_and_cyclic_dependencies():
    registry = load_registry()
    unknown = deepcopy(registry)
    unknown["families"][0]["depends_on"] = ["E99"]
    with pytest.raises(ValueError, match="unknown dependencies"):
        validate_registry(unknown)

    cyclic = deepcopy(registry)
    cyclic["families"][0]["depends_on"] = ["E11"]
    with pytest.raises(ValueError, match="Dependency cycle"):
        validate_registry(cyclic)


def test_query_reranker_batch_size_defaults_to_measured_control():
    args = parser().parse_args(["experiment", "query", "--strategy", "original"])
    assert args.rerank_batch_size == 64
    assert args.offset == 0
