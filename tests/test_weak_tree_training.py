"""Small labeled cases for weighted greedy weak-tree search."""

from __future__ import annotations

import unittest

import numpy as np

from src.weak_tree import PixelDifferenceFeature
from src.weak_tree_training import (
    fit_random_depth2_tree,
    sample_random_features,
    search_depth2_tree,
)


ROOT = PixelDifferenceFeature(0, (0, 0), (1, 0))
CHILD = PixelDifferenceFeature(0, (0, 1), (1, 1))


def windows_for(root_values: list[int], child_values: list[int]) -> np.ndarray:
    windows = np.zeros((len(root_values), 11, 24, 24), dtype=np.uint8)
    for i, (root, child) in enumerate(zip(root_values, child_values, strict=True)):
        for y, difference in enumerate((root, child)):
            windows[i, 0, y, 0] = max(difference, 0)
            windows[i, 0, y, 1] = max(-difference, 0)
    return windows


class WeakTreeTrainingTests(unittest.TestCase):
    def test_greedy_search_fits_four_distinct_leaves(self) -> None:
        windows = windows_for([-1, -1, 1, 1], [-1, 1, -1, 1])
        labels = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = search_depth2_tree(windows, labels, np.ones(4), (ROOT,), (CHILD,))
        self.assertEqual(result.weighted_error, 0.0)
        self.assertEqual(result.tree.root.threshold, -1)
        self.assertEqual(result.tree.left.threshold, -1)
        self.assertEqual(result.tree.right.threshold, -1)
        scores = result.tree.predict_scores(windows)
        np.testing.assert_array_equal(scores > 0, labels == 1)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_weights_control_majority_and_error(self) -> None:
        windows = windows_for([0, 0], [0, 0])
        labels = np.array([0, 1], dtype=np.uint8)
        result = search_depth2_tree(windows, labels, np.array([1.0, 3.0]), (ROOT,), (CHILD,))
        self.assertAlmostEqual(result.weighted_error, 0.25)
        self.assertTrue(np.all(result.tree.predict_scores(windows) > 0))

    def test_weights_change_the_selected_threshold(self) -> None:
        windows = windows_for([-2, -1, 1, 2], [0, 0, 0, 0])
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)
        uniform = search_depth2_tree(windows, labels, np.ones(4), (ROOT,), (CHILD,))
        weighted = search_depth2_tree(windows, labels, np.array([1, 5, 5, 1]), (ROOT,), (CHILD,))
        self.assertEqual(uniform.tree.root.threshold, -2)
        self.assertEqual(weighted.tree.root.threshold, -1)

    def test_random_candidates_are_reproducible_and_distinct(self) -> None:
        first = sample_random_features(np.random.default_rng(42), 100)
        second = sample_random_features(np.random.default_rng(42), 100)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 100)
        windows = windows_for([-1, 1], [-1, 1])
        labels = np.array([0, 1], dtype=np.uint8)
        a = fit_random_depth2_tree(windows, labels, np.ones(2), seed=42, root_candidates=5, child_candidates=5)
        b = fit_random_depth2_tree(windows, labels, np.ones(2), seed=42, root_candidates=5, child_candidates=5)
        self.assertEqual(a, b)

    def test_invalid_training_inputs(self) -> None:
        windows = windows_for([-1, 1], [-1, 1])
        labels = np.array([0, 1], dtype=np.uint8)
        with self.assertRaises(ValueError):
            search_depth2_tree(windows, labels, np.array([1.0, -1.0]), (ROOT,), (CHILD,))
        with self.assertRaises(ValueError):
            search_depth2_tree(windows, np.array([0, 2]), np.ones(2), (ROOT,), (CHILD,))
        with self.assertRaises(ValueError):
            search_depth2_tree(windows, labels, np.ones(2), (), (CHILD,))
        with self.assertRaises(ValueError):
            sample_random_features(np.random.default_rng(1), 0)


if __name__ == "__main__":
    unittest.main()
