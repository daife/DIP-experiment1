import unittest

import numpy as np

from src.candidate_verifier import features, hog


class CandidateVerifierFeaturesTest(unittest.TestCase):
    def test_constant_and_gradient_patches_are_finite_and_distinct(self):
        constant = np.full((24, 24), 128, dtype=np.uint8)
        gradient = np.tile(np.arange(24, dtype=np.uint8) * 10, (24, 1))
        a, b = hog(constant), hog(gradient)
        self.assertEqual(a.shape, (324,))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        self.assertTrue(np.all(a == 0))
        self.assertGreater(np.linalg.norm(b), 0)

    def test_empty_boxes_have_model_feature_width(self):
        gray = np.zeros((24, 24), dtype=np.uint8)
        self.assertEqual(features(gray, np.empty((0, 4), dtype=np.int32)).shape, (0, 324))


if __name__ == "__main__":
    unittest.main()
