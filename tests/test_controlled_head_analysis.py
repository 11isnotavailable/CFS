import numpy as np

from scripts.analyze_controlled_head_pilot import top_fraction_mask


def test_top_fraction_mask_selects_largest_drops_per_sample():
    drops = np.array([[0.1, 0.4, 0.2, 0.3], [4.0, 1.0, 3.0, 2.0]])
    mask = top_fraction_mask(drops, 0.5)
    assert np.array_equal(mask[0], [False, True, False, True])
    assert np.array_equal(mask[1], [True, False, True, False])
