import unittest

from scripts.prepare_detection_dataset import near_face, square_box


class DetectionCropTests(unittest.TestCase):
    def test_positive_square_preserves_center_and_margin(self):
        box = square_box([10, 20, 50, 40])
        self.assertEqual(box[2] - box[0], box[3] - box[1])
        self.assertLessEqual(box[0], 10)
        self.assertLessEqual(box[1], 20)
        self.assertGreaterEqual(box[2], 50)
        self.assertGreaterEqual(box[3], 40)

    def test_negative_rejects_tiny_face_and_nearby_window(self):
        faces = [[100, 100, 110, 110]]
        self.assertTrue(near_face([0, 0, 200, 200], faces))
        self.assertTrue(near_face([90, 90, 100, 100], faces))
        self.assertFalse(near_face([0, 0, 80, 80], faces))


if __name__ == "__main__":
    unittest.main()
