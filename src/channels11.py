"""Integer implementation of the 11 hand-crafted face features.

The layout and boundary convention follow ``cpu-face-hpc/face_frontal.py``:
locations for which the full source neighbourhood is unavailable remain zero.
No convolution routine is used; every assignment exposes its source pixels.
"""

from __future__ import annotations

import numpy as np


NUM_CHANNELS = 11


def compute_11_channels(gray: np.ndarray) -> list[np.ndarray]:
    """Return the 11 ``uint8`` feature channels for a grayscale image.

    Args:
        gray: A two-dimensional ``numpy.ndarray`` with dtype ``uint8``.

    Returns:
        Eleven independent arrays with the same shape as ``gray``. Invalid
        right/bottom boundary locations are consistently left at zero.

    Raises:
        TypeError: If ``gray`` is not a NumPy ``uint8`` array.
        ValueError: If ``gray`` is not a non-empty two-dimensional image.
    """
    if not isinstance(gray, np.ndarray):
        raise TypeError("gray must be a numpy.ndarray")
    if gray.dtype != np.uint8:
        raise TypeError(f"gray must have dtype uint8, got {gray.dtype}")
    if gray.ndim != 2:
        raise ValueError(f"gray must be two-dimensional, got shape {gray.shape}")
    if gray.size == 0:
        raise ValueError("gray must not be empty")

    height, width = gray.shape
    channels = [np.zeros_like(gray) for _ in range(NUM_CHANNELS)]
    channels[0] = gray.copy()

    # C1[y,x]: rounded, pairwise 2x2 mean. int16 prevents uint8 overflow.
    if height > 1 and width > 1:
        source = gray.astype(np.int16)
        left_column = (
            source[:-1, :-1] + source[1:, :-1] + 1
        ) // 2
        right_column = (
            source[:-1, 1:] + source[1:, 1:] + 1
        ) // 2
        channels[1][:-1, :-1] = (
            left_column + right_column + 1
        ) // 2

    # The reference implementation reserves three bottom/right boundary
    # pixels for C2-C6, even though each formula spans two source pixels.
    short_height = height - 3
    short_width = width - 3
    if short_height > 0 and short_width > 0:
        c1 = channels[1].astype(np.int16)
        p00 = c1[:short_height, :short_width]
        p02 = c1[:short_height, 2 : 2 + short_width]
        p20 = c1[2 : 2 + short_height, :short_width]
        p22 = c1[2 : 2 + short_height, 2 : 2 + short_width]

        # C2: mean of four C1 samples separated by two pixels. The asymmetric
        # +1 placement exactly matches cpu-face-hpc's integer rounding order.
        channels[2][:short_height, :short_width] = (
            ((p00 + p20) // 2) + ((p22 + p02 + 1) // 2) + 1
        ) // 2

        # C3-C6: horizontal, vertical, main diagonal, anti-diagonal.
        channels[3][:short_height, :short_width] = (p02 - p00 + 255) // 2
        channels[4][:short_height, :short_width] = (p20 - p00 + 255) // 2
        channels[5][:short_height, :short_width] = (p22 - p00 + 255) // 2
        channels[6][:short_height, :short_width] = (p20 - p02 + 255) // 2

    # C7-C10 use the same directions on C2 with a four-pixel displacement.
    # Seven bottom/right boundary pixels remain zero, matching the reference.
    long_height = height - 7
    long_width = width - 7
    if long_height > 0 and long_width > 0:
        c2 = channels[2].astype(np.int16)
        q00 = c2[:long_height, :long_width]
        q04 = c2[:long_height, 4 : 4 + long_width]
        q40 = c2[4 : 4 + long_height, :long_width]
        q44 = c2[4 : 4 + long_height, 4 : 4 + long_width]

        channels[7][:long_height, :long_width] = (q04 - q00 + 255) // 2
        channels[8][:long_height, :long_width] = (q40 - q00 + 255) // 2
        channels[9][:long_height, :long_width] = (q44 - q00 + 255) // 2
        channels[10][:long_height, :long_width] = (q40 - q04 + 255) // 2

    return channels
