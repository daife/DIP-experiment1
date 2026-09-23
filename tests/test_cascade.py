"""AdaBoost stage and Cascade behavior on controlled feature windows."""

from __future__ import annotations

import unittest

import numpy as np

from src.cascade import (
    Cascade, CascadeStage, cascade_from_dict, cascade_to_dict,
    fit_stage, stage_statistics, threshold_for_recall,
)
from src.weak_tree import Depth2WeakTree, PixelDifferenceFeature, TreeNode


def windows(values: list[int]) -> np.ndarray:
    batch = np.zeros((len(values), 11, 24, 24), dtype=np.uint8)
    for i, value in enumerate(values):
        batch[i, 0, 0, 0] = max(value, 0)
        batch[i, 0, 0, 1] = max(-value, 0)
    return batch


class CascadeTests(unittest.TestCase):
    def test_recall_threshold_handles_ties(self) -> None:
        self.assertEqual(threshold_for_recall(np.array([1., 1., 2., 3.]), 0.75), 1.)
        self.assertEqual(threshold_for_recall(np.array([1., 2., 3., 4.]), 0.5), 3.)

    def test_stage_fit_and_model_round_trip(self) -> None:
        training = windows([-2, -1, 1, 2])
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        stage = fit_stage(training, labels, training[labels == 1], seed=12,
                          num_trees=2, root_candidates=200, child_candidates=10)
        model = Cascade((stage,))
        recovered = cascade_from_dict(cascade_to_dict(model))
        np.testing.assert_array_equal(model.evaluate(training)[0], recovered.evaluate(training)[0])
        np.testing.assert_allclose(model.evaluate(training)[1], recovered.evaluate(training)[1])

    def test_early_rejection_and_stage_counts(self) -> None:
        feature = PixelDifferenceFeature(0, (0, 0), (1, 0))
        node = TreeNode(feature, 0)
        tree = Depth2WeakTree(node, node, node, (-1., -1., 1., 1.))
        model = Cascade((CascadeStage((tree,), 0.), CascadeStage((tree,), 1.5)))
        batch = windows([-1, 1])
        labels = np.array([0, 1])
        passed, scores, trace = model.evaluate(batch)
        np.testing.assert_array_equal(passed, [False, False])
        self.assertEqual(trace, [{"entering": 2, "passing": 1}, {"entering": 1, "passing": 0}])
        self.assertTrue(np.isfinite(scores[1]))
        report = stage_statistics(model, batch, labels)
        self.assertEqual(report[0]["negative_rejected"], 1)
        self.assertEqual(report[1]["positive_passing"], 0)


if __name__ == "__main__":
    unittest.main()
