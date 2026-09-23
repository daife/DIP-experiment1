"""Acceptance tests for the integer 11-channel implementation."""

from __future__ import annotations

import unittest

import numpy as np

from src.channels11 import compute_11_channels


def scalar_reference(gray: np.ndarray) -> list[np.ndarray]:
    """Deliberately scalar transcription used to audit the vectorized code."""
    height, width = gray.shape
    result = [np.zeros_like(gray) for _ in range(11)]
    result[0] = gray.copy()

    for y in range(height - 1):
        for x in range(width - 1):
            a = (int(gray[y, x]) + int(gray[y + 1, x]) + 1) // 2
            b = (int(gray[y, x + 1]) + int(gray[y + 1, x + 1]) + 1) // 2
            result[1][y, x] = (a + b + 1) // 2

    for y in range(max(0, height - 3)):
        for x in range(max(0, width - 3)):
            p00 = int(result[1][y, x])
            p02 = int(result[1][y, x + 2])
            p20 = int(result[1][y + 2, x])
            p22 = int(result[1][y + 2, x + 2])
            result[2][y, x] = ((p00 + p20) // 2 + (p22 + p02 + 1) // 2 + 1) // 2
            result[3][y, x] = (p02 - p00 + 255) // 2
            result[4][y, x] = (p20 - p00 + 255) // 2
            result[5][y, x] = (p22 - p00 + 255) // 2
            result[6][y, x] = (p20 - p02 + 255) // 2

    for y in range(max(0, height - 7)):
        for x in range(max(0, width - 7)):
            q00 = int(result[2][y, x])
            q04 = int(result[2][y, x + 4])
            q40 = int(result[2][y + 4, x])
            q44 = int(result[2][y + 4, x + 4])
            result[7][y, x] = (q04 - q00 + 255) // 2
            result[8][y, x] = (q40 - q00 + 255) // 2
            result[9][y, x] = (q44 - q00 + 255) // 2
            result[10][y, x] = (q40 - q04 + 255) // 2
    return result


class Compute11ChannelsTests(unittest.TestCase):
    def setUp(self) -> None:
        # An asymmetric ramp makes every direction distinguishable.
        self.gray = np.array(
            [
                [0, 5, 12, 21, 32, 45, 60, 77, 96],
                [9, 15, 23, 33, 45, 59, 75, 93, 113],
                [20, 27, 36, 47, 60, 75, 92, 111, 132],
                [33, 41, 51, 63, 77, 93, 111, 131, 153],
                [48, 57, 68, 81, 96, 113, 132, 153, 176],
                [65, 75, 87, 101, 117, 135, 155, 177, 201],
                [84, 95, 108, 123, 140, 159, 180, 203, 228],
                [105, 117, 131, 147, 165, 185, 207, 231, 255],
                [128, 141, 156, 173, 192, 213, 236, 250, 254],
            ],
            dtype=np.uint8,
        )

    def test_count_shape_dtype_and_independence(self) -> None:
        channels = compute_11_channels(self.gray)
        self.assertEqual(len(channels), 11)
        for channel in channels:
            self.assertEqual(channel.shape, self.gray.shape)
            self.assertEqual(channel.dtype, np.uint8)
        self.assertFalse(np.shares_memory(channels[0], self.gray))
        self.assertEqual(len({id(channel) for channel in channels}), 11)

    def test_small_matrix_matches_each_scalar_formula(self) -> None:
        actual = compute_11_channels(self.gray)
        expected = scalar_reference(self.gray)
        for index, (actual_channel, expected_channel) in enumerate(
            zip(actual, expected, strict=True)
        ):
            np.testing.assert_array_equal(
                actual_channel,
                expected_channel,
                err_msg=f"channel {index} differs from scalar formula",
            )

    def test_spot_values_cover_all_formula_families(self) -> None:
        channels = compute_11_channels(self.gray)
        self.assertEqual(int(channels[0][0, 0]), 0)
        self.assertEqual(int(channels[1][0, 0]), 8)
        self.assertEqual(int(channels[2][0, 0]), 28)
        self.assertEqual([int(channels[i][0, 0]) for i in range(3, 7)], [135, 139, 148, 131])
        self.assertEqual([int(channels[i][0, 0]) for i in range(7, 11)], [152, 160, 193, 135])

    def test_reference_zero_boundaries(self) -> None:
        channels = compute_11_channels(self.gray)
        self.assertTrue(np.all(channels[1][-1, :] == 0))
        self.assertTrue(np.all(channels[1][:, -1] == 0))
        for index in range(2, 7):
            self.assertTrue(np.all(channels[index][-3:, :] == 0))
            self.assertTrue(np.all(channels[index][:, -3:] == 0))
        for index in range(7, 11):
            self.assertTrue(np.all(channels[index][-7:, :] == 0))
            self.assertTrue(np.all(channels[index][:, -7:] == 0))

    def test_tiny_images_are_supported(self) -> None:
        for shape in ((1, 1), (2, 2), (4, 4), (7, 7)):
            channels = compute_11_channels(np.zeros(shape, dtype=np.uint8))
            self.assertEqual(len(channels), 11)
            self.assertTrue(all(channel.shape == shape for channel in channels))

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            compute_11_channels(np.zeros((9, 9), dtype=np.float32))
        with self.assertRaises(ValueError):
            compute_11_channels(np.zeros((9, 9, 1), dtype=np.uint8))
        with self.assertRaises(ValueError):
            compute_11_channels(np.zeros((0, 9), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
