"""Synthetic sanity checks for CFS graph capacity proxies."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_functional_capacity import (
    exact_minimum_cover,
    spectral_effective_rank,
    subset_table,
)


class FunctionalCapacityTests(unittest.TestCase):
    def test_identity_graph_has_full_capacity(self) -> None:
        for heads in (3, 6, 12):
            scores = np.zeros((1, heads, heads), dtype=np.float64)
            valid = np.ones((1, heads), dtype=bool)
            selections, cardinality = subset_table(heads)
            cover = exact_minimum_cover(
                scores, valid, 0.5, selections, cardinality
            )
            rank = spectral_effective_rank(scores, valid)
            self.assertEqual(int(cover[0]), heads)
            self.assertAlmostEqual(float(rank[0]), heads, places=6)

    def test_fully_interchangeable_graph_has_unit_capacity(self) -> None:
        for heads in (3, 6, 12):
            scores = np.ones((1, heads, heads), dtype=np.float64)
            valid = np.ones((1, heads), dtype=bool)
            selections, cardinality = subset_table(heads)
            cover = exact_minimum_cover(
                scores, valid, 0.5, selections, cardinality
            )
            rank = spectral_effective_rank(scores, valid)
            self.assertEqual(int(cover[0]), 1)
            self.assertAlmostEqual(float(rank[0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
