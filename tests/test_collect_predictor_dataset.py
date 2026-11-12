import numpy as np

from scripts.collect_predictor_dataset import evenly_subsample_indices


def test_evenly_subsample_indices_preserves_full_range():
    values = np.arange(512)
    selected = evenly_subsample_indices(values, 128)
    assert len(selected) == 128
    assert selected[0] == 0
    assert selected[-1] == 511
    assert np.all(np.diff(selected) > 0)


def test_evenly_subsample_indices_keeps_short_input():
    values = np.array([3, 8, 13])
    assert np.array_equal(evenly_subsample_indices(values, 10), values)
