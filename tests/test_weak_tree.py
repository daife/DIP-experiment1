"""Branch, threshold, and signed-arithmetic tests for depth-2 inference."""

from __future__ import annotations

import unittest

import numpy as np

from src.weak_tree import Depth2WeakTree, PixelDifferenceFeature, TreeNode


class WeakTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        # All three nodes read independent pixel pairs in channel zero.
        self.tree = Depth2WeakTree(
            root=TreeNode(PixelDifferenceFeature(0, (0, 0), (1, 0)), 0),
            left=TreeNode(PixelDifferenceFeature(0, (0, 1), (1, 1)), -1),
            right=TreeNode(PixelDifferenceFeature(0, (0, 2), (1, 2)), 1),
            leaf_scores=(-2.0, 0.5, 1.25, 3.0),
        )

    @staticmethod
    def window(root: int, left: int, right: int) -> np.ndarray:
        result = np.zeros((11, 24, 24), dtype=np.uint8)
        for y, difference in enumerate((root, left, right)):
            result[0, y, 0] = max(difference, 0)
            result[0, y, 1] = max(-difference, 0)
        return result

    def test_signed_difference_uses_xy_coordinates_and_int16(self) -> None:
        window = np.zeros((11, 24, 24), dtype=np.uint8)
        window[7, 3, 2] = 0
        window[7, 5, 4] = 255
        feature = PixelDifferenceFeature(7, (2, 3), (4, 5))
        self.assertEqual(int(feature.evaluate(window)), -255)
        batch = np.stack((window, np.zeros_like(window)))
        np.testing.assert_array_equal(feature.evaluate(batch), np.array([-255, 0], dtype=np.int16))

    def test_four_leaves_and_threshold_ties(self) -> None:
        cases = (
            (self.window(0, -1, 100), -2.0),
            (self.window(-1, 0, -100), 0.5),
            (self.window(1, -100, 1), 1.25),
            (self.window(2, 100, 2), 3.0),
        )
        batch = np.stack([window for window, _ in cases])
        expected = [score for _, score in cases]
        self.assertEqual([self.tree.predict_scores(window) for window, _ in cases], expected)
        np.testing.assert_array_equal(self.tree.predict_scores(batch), expected)
        self.assertEqual(self.tree.predict_scores(batch[:0]).shape, (0,))

    def test_invalid_feature_tree_and_input(self) -> None:
        with self.assertRaises(ValueError):
            PixelDifferenceFeature(11, (0, 0), (1, 0))
        with self.assertRaises(ValueError):
            PixelDifferenceFeature(0, (24, 0), (1, 0))
        with self.assertRaises(ValueError):
            PixelDifferenceFeature(0, (0, 0), (0, 0))
        with self.assertRaises(ValueError):
            TreeNode(PixelDifferenceFeature(0, (0, 0), (1, 0)), 256)
        with self.assertRaises(ValueError):
            Depth2WeakTree(self.tree.root, self.tree.left, self.tree.right, (0, 1, np.nan, 3))
        with self.assertRaises(TypeError):
            self.tree.predict_scores(np.zeros((11, 24, 24), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.tree.predict_scores(np.zeros((11, 23, 24), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
