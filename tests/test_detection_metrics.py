import unittest

import numpy as np

from src.detection_metrics import match_detections


class DetectionMetricsTests(unittest.TestCase):
    def test_duplicate_and_missed_face(self):
        gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        result = match_detections(boxes, np.array([2., 1.]), gt)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["f1"], 0.5)

    def test_empty_predictions(self):
        result = match_detections(np.empty((0, 4)), np.empty(0), np.array([[0, 0, 4, 4]]))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 1))
