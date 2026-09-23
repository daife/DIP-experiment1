"""Full-page feature lookup must match 24x24 training-crop semantics."""

from __future__ import annotations

import unittest

import numpy as np

from src.cascade import Cascade, CascadeStage
from src.channels11 import compute_11_channels
from src.page_scan import _feature_values, scan_page
from src.weak_tree import Depth2WeakTree, PixelDifferenceFeature, TreeNode


class PageScanTests(unittest.TestCase):
    def test_channel_edges_match_local_crop(self) -> None:
        gray = np.random.default_rng(7).integers(0, 256, (34, 39), dtype=np.uint8)
        full = compute_11_channels(gray)
        for channel in range(11):
            feature = PixelDifferenceFeature(channel, (23, 23), (0, 0))
            actual = _feature_values(full, feature, np.array([0, 7]), np.array([0, 5]))
            expected = []
            for x, y in ((0, 0), (7, 5)):
                local = compute_11_channels(gray[y:y + 24, x:x + 24])
                expected.append(int(local[channel][23, 23]) - int(local[channel][0, 0]))
            np.testing.assert_array_equal(actual, expected)

    def test_scan_agrees_with_crop_inference(self) -> None:
        gray = np.random.default_rng(9).integers(0, 256, (29, 30), dtype=np.uint8)
        feature = PixelDifferenceFeature(3, (20, 20), (21, 21))
        node = TreeNode(feature, 0)
        tree = Depth2WeakTree(node, node, node, (1., 1., -1., -1.))
        model = Cascade((CascadeStage((tree,), -2.),))
        boxes, scores = scan_page(gray, model, step=2)
        crops = np.stack([compute_11_channels(gray[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes])
        expected = model.evaluate(crops)[1]
        np.testing.assert_array_equal(scores, expected)


if __name__ == "__main__":
    unittest.main()
