"""Scan fixed-size windows on a resized page with a trained Cascade."""

from __future__ import annotations

import numpy as np

from .cascade import Cascade
from .channels11 import compute_11_channels
from .weak_tree import Depth2WeakTree, PixelDifferenceFeature


def _feature_values(
    channels: list[np.ndarray], feature: PixelDifferenceFeature,
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    x1, y1 = feature.p1
    x2, y2 = feature.p2
    # Channel boundaries are relative to each 24x24 training crop. Values
    # there are zero even when a full-page channel has neighboring pixels.
    invalid = 0 if feature.channel == 0 else 1 if feature.channel == 1 else 3 if feature.channel <= 6 else 7
    limit = 24 - invalid
    first = channels[feature.channel][ys + y1, xs + x1].astype(np.int16) if x1 < limit and y1 < limit else np.zeros(len(xs), dtype=np.int16)
    second = channels[feature.channel][ys + y2, xs + x2].astype(np.int16) if x2 < limit and y2 < limit else np.zeros(len(xs), dtype=np.int16)
    return first - second


def _tree_scores(
    channels: list[np.ndarray], tree: Depth2WeakTree,
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    root_left = _feature_values(channels, tree.root.feature, xs, ys) <= tree.root.threshold
    left_left = _feature_values(channels, tree.left.feature, xs, ys) <= tree.left.threshold
    right_left = _feature_values(channels, tree.right.feature, xs, ys) <= tree.right.threshold
    indices = np.where(root_left, np.where(left_left, 0, 1), np.where(right_left, 2, 3))
    return np.asarray(tree.leaf_scores)[indices]


def scan_page(gray: np.ndarray, cascade: Cascade, *, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return passing ``(x1,y1,x2,y2)`` windows and final-stage scores."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a uint8 image")
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    height, width = gray.shape
    if min(height, width) < 24:
        return np.empty((0, 4), dtype=np.int32), np.empty(0, dtype=np.float64)
    channels = compute_11_channels(gray)
    grid_x, grid_y = np.meshgrid(np.arange(0, width - 23, step), np.arange(0, height - 23, step))
    xs, ys = grid_x.ravel(), grid_y.ravel()
    scores = np.empty(0, dtype=np.float64)
    for stage in cascade.stages:
        scores = np.zeros(len(xs), dtype=np.float64)
        for tree in stage.trees:
            scores += stage.learning_rate * _tree_scores(channels, tree, xs, ys)
        keep = scores >= stage.threshold
        xs, ys, scores = xs[keep], ys[keep], scores[keep]
        if not len(xs):
            break
    boxes = np.stack((xs, ys, xs + 24, ys + 24), axis=1).astype(np.int32)
    return boxes, scores
