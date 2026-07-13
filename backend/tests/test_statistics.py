import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from statistics import confusion_matrix, paired_bootstrap, weighted_kappa


def test_paired_statistics():
    assert paired_bootstrap([0, 0, 1], [1, 1, 1], resamples=100)["mean_delta_right_minus_left"] > 0
    assert weighted_kappa([0, 1, 2], [0, 1, 2]) == 1.0
    assert confusion_matrix([0, 1, 2], [0, 2, 2]) == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
