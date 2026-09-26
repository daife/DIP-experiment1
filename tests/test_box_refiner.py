"""Geometric clipping/empty-output checks for optional proposal refinement."""
import unittest
import numpy as np
from src.box_refiner import transform_boxes


class BoxRefinerTests(unittest.TestCase):
    def test_center_and_size_deltas(self):
        boxes,valid=transform_boxes([[10,10,30,30]],[[.25,-.5,np.log(2),0]],100,100)
        np.testing.assert_array_equal(boxes,[[5,0,45,20]])
        self.assertTrue(valid[0])

    def test_clipped_collapsed_box_is_invalid(self):
        boxes,valid=transform_boxes([[0,0,10,10]],[[-100,-100,0,0]],100,100)
        self.assertFalse(valid[0])
        self.assertTrue((boxes>=0).all())

    def test_empty_and_nonfinite(self):
        boxes,valid=transform_boxes(np.empty((0,4)),np.empty((0,4)),100,100)
        self.assertEqual(boxes.shape,(0,4));self.assertEqual(valid.shape,(0,))
        with self.assertRaises(ValueError):transform_boxes([[0,0,10,10]],[[np.nan,0,0,0]],100,100)


if __name__=='__main__':unittest.main()
