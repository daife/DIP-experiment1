"""Pixel-difference features and depth-2 weak-tree inference.

Inputs are the cached 11-channel, 24x24 ``uint8`` windows. A node goes left
when its signed pixel difference is less than or equal to its threshold.
Leaf order is (root-left/child-left, root-left/child-right,
root-right/child-left, root-right/child-right).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels11 import NUM_CHANNELS


WINDOW_SIZE = 24


def _check_windows(windows: np.ndarray) -> None:
    if not isinstance(windows, np.ndarray):
        raise TypeError("windows must be a numpy.ndarray")
    if windows.dtype != np.uint8:
        raise TypeError(f"windows must have dtype uint8, got {windows.dtype}")
    if windows.ndim not in (3, 4):
        raise ValueError("windows must have shape (11, 24, 24) or (N, 11, 24, 24)")
    if windows.shape[-3:] != (NUM_CHANNELS, WINDOW_SIZE, WINDOW_SIZE):
        raise ValueError("windows must have shape (11, 24, 24) or (N, 11, 24, 24)")


@dataclass(frozen=True, slots=True)
class PixelDifferenceFeature:
    """Signed difference ``channel[y1, x1] - channel[y2, x2]``."""

    channel: int
    p1: tuple[int, int]  # (x, y)
    p2: tuple[int, int]  # (x, y)

    def __post_init__(self) -> None:
        if type(self.channel) is not int or not 0 <= self.channel < NUM_CHANNELS:
            raise ValueError(f"channel must be an integer in [0, {NUM_CHANNELS - 1}]")
        for name, point in (("p1", self.p1), ("p2", self.p2)):
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or any(type(v) is not int or not 0 <= v < WINDOW_SIZE for v in point)
            ):
                raise ValueError(f"{name} must be an (x, y) integer pair in [0, 23]")
        if self.p1 == self.p2:
            raise ValueError("p1 and p2 must differ")

    def evaluate(self, windows: np.ndarray) -> np.int16 | np.ndarray:
        """Return a signed scalar for one window or an ``int16`` batch vector."""
        _check_windows(windows)
        x1, y1 = self.p1
        x2, y2 = self.p2
        if windows.ndim == 3:
            return np.int16(
                int(windows[self.channel, y1, x1])
                - int(windows[self.channel, y2, x2])
            )
        return (
            windows[:, self.channel, y1, x1].astype(np.int16)
            - windows[:, self.channel, y2, x2].astype(np.int16)
        )


@dataclass(frozen=True, slots=True)
class TreeNode:
    feature: PixelDifferenceFeature
    threshold: int

    def __post_init__(self) -> None:
        if type(self.threshold) is not int or not -255 <= self.threshold <= 255:
            raise ValueError("threshold must be an integer in [-255, 255]")


@dataclass(frozen=True, slots=True)
class Depth2WeakTree:
    """A root, one child per root branch, and four real-valued leaf scores."""

    root: TreeNode
    left: TreeNode
    right: TreeNode
    leaf_scores: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not all(isinstance(node, TreeNode) for node in (self.root, self.left, self.right)):
            raise TypeError("root, left, and right must be TreeNode instances")
        if len(self.leaf_scores) != 4 or not all(np.isfinite(v) for v in self.leaf_scores):
            raise ValueError("leaf_scores must contain four finite numbers")

    def predict_scores(self, windows: np.ndarray) -> float | np.ndarray:
        """Evaluate one window or a batch; positive scores favor a face."""
        _check_windows(windows)
        root_left = self.root.feature.evaluate(windows) <= self.root.threshold
        left_left = self.left.feature.evaluate(windows) <= self.left.threshold
        right_left = self.right.feature.evaluate(windows) <= self.right.threshold
        if windows.ndim == 3:
            index = (0 if root_left else 2) + (0 if (left_left if root_left else right_left) else 1)
            return float(self.leaf_scores[index])
        indices = np.where(root_left, np.where(left_left, 0, 1), np.where(right_left, 2, 3))
        return np.asarray(self.leaf_scores, dtype=np.float64)[indices]
