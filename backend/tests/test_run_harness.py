import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from run_bioasq_end_to_end import median


def test_harness_median_handles_even_and_odd_samples():
    assert median([3, 1, 2]) == 2
    assert median([1, 3]) == 2
