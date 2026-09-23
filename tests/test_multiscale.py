"""Pyramid geometry, coordinate restoration, and duplicate suppression."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.multiscale import PyramidConfig, detect_multiscale, nms, pyramid_sizes


class MultiscaleTests(unittest.TestCase):
    def test_pyramid_stops_at_24_and_has_unique_sizes(self) -> None:
        self.assertEqual(pyramid_sizes(48, 36, 1.2), [(48, 36), (40, 30), (33, 25)])
        self.assertEqual(pyramid_sizes(23, 36, 1.2), [])

    def test_nms_keeps_highest_score_and_separate_face(self) -> None:
        boxes = np.array([[0, 0, 24, 24], [1, 1, 25, 25], [40, 40, 64, 64]], dtype=np.int32)
        scores = np.array([0.8, 0.9, 0.7])
        np.testing.assert_array_equal(nms(boxes, scores, 0.3), [1, 2])
        np.testing.assert_array_equal(nms(np.empty((0, 4)), np.empty(0)), [])

    def test_mapping_and_window_counts(self) -> None:
        gray = np.zeros((36, 48), dtype=np.uint8)
        results = [(np.array([[2, 3, 26, 27]], dtype=np.int32), np.array([1.0])),
                   (np.array([[2, 3, 26, 27]], dtype=np.int32), np.array([2.0])),
                   (np.empty((0, 4), dtype=np.int32), np.empty(0))]
        with patch("src.multiscale.scan_page", side_effect=results) as scan:
            boxes, scores, layers = detect_multiscale(gray, object(), PyramidConfig(1.2, 2, 0.3))
        self.assertEqual(scan.call_count, 3)
        self.assertTrue(all(call.kwargs["step"] == 2 for call in scan.call_args_list))
        self.assertEqual([layer["windows"] for layer in layers], [13 * 7, 9 * 4, 5 * 1])
        np.testing.assert_array_equal(boxes, [[2, 4, 31, 32]])
        np.testing.assert_array_equal(scores, [2.0])

    def test_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            PyramidConfig(scale_factor=1)
        with self.assertRaises(ValueError):
            PyramidConfig(step=0)


if __name__ == "__main__":
    unittest.main()
