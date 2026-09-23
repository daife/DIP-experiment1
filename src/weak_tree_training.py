"""Weighted greedy search for one depth-2 pixel-difference weak tree.

This module trains only a single tree. AdaBoost weight updates, stage
thresholds, and Cascade construction belong to later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .channels11 import NUM_CHANNELS
from .weak_tree import (
    WINDOW_SIZE,
    Depth2WeakTree,
    PixelDifferenceFeature,
    TreeNode,
    _check_windows,
)


@dataclass(frozen=True, slots=True)
class TreeSearchResult:
    tree: Depth2WeakTree
    weighted_error: float


def sample_random_features(
    rng: np.random.Generator, count: int
) -> tuple[PixelDifferenceFeature, ...]:
    """Sample distinct feature definitions from the fixed 24x24 window."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    maximum = NUM_CHANNELS * WINDOW_SIZE**2 * (WINDOW_SIZE**2 - 1)
    if count > maximum:
        raise ValueError(f"count cannot exceed {maximum}")
    features: list[PixelDifferenceFeature] = []
    seen: set[PixelDifferenceFeature] = set()
    while len(features) < count:
        channel = int(rng.integers(NUM_CHANNELS))
        first = int(rng.integers(WINDOW_SIZE**2))
        second = int(rng.integers(WINDOW_SIZE**2 - 1))
        if second >= first:
            second += 1
        feature = PixelDifferenceFeature(
            channel,
            (first % WINDOW_SIZE, first // WINDOW_SIZE),
            (second % WINDOW_SIZE, second // WINDOW_SIZE),
        )
        if feature not in seen:
            seen.add(feature)
            features.append(feature)
    return tuple(features)


def _best_split(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> tuple[int, float]:
    """Find the threshold minimizing weighted majority-vote leaf error."""
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return 255, 0.0
    order = selected[np.argsort(values[selected], kind="stable")]
    sorted_values = values[order]
    positive = weights[order] * labels[order]
    negative = weights[order] * (1 - labels[order])
    cumulative_positive = np.cumsum(positive)
    cumulative_negative = np.cumsum(negative)
    total_positive = cumulative_positive[-1]
    total_negative = cumulative_negative[-1]
    ends = np.r_[np.flatnonzero(np.diff(sorted_values)), len(order) - 1]
    left_positive = cumulative_positive[ends]
    left_negative = cumulative_negative[ends]
    errors = (
        np.minimum(left_positive, left_negative)
        + np.minimum(total_positive - left_positive, total_negative - left_negative)
    )
    best = int(np.argmin(errors))
    return int(sorted_values[ends[best]]), float(errors[best])


def _leaf_score(positive: float, negative: float) -> float:
    # A finite log-odds score keeps pure and empty leaves representable.
    return float(0.5 * np.log((positive + 1e-12) / (negative + 1e-12)))


def _validate_training_data(
    windows: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _check_windows(windows)
    if windows.ndim != 4 or len(windows) == 0:
        raise ValueError("windows must be a nonempty batch of shape (N, 11, 24, 24)")
    labels = np.asarray(labels)
    sample_weight = np.asarray(sample_weight)
    if labels.shape != (len(windows),) or not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be a length-N vector containing only 0 and 1")
    if sample_weight.shape != (len(windows),):
        raise ValueError("sample_weight must be a length-N vector")
    if not np.issubdtype(sample_weight.dtype, np.number):
        raise ValueError("sample_weight must be numeric")
    weights = sample_weight.astype(np.float64)
    weight_sum = weights.sum()
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("sample_weight must be finite, nonnegative, and have positive sum")
    return labels.astype(np.int8), weights / weight_sum


def search_depth2_tree(
    windows: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    root_features: Sequence[PixelDifferenceFeature],
    child_features: Sequence[PixelDifferenceFeature],
) -> TreeSearchResult:
    """Search root thresholds, then both child splits, for each root feature.

    A threshold sends values ``<= threshold`` left. Each node threshold is
    chosen by weighted majority-vote error; the best complete tree wins.
    Weights are normalized, so the returned error lies in ``[0, 1]``.
    """
    labels, weights = _validate_training_data(windows, labels, sample_weight)
    if not root_features or not child_features:
        raise ValueError("root_features and child_features must be nonempty")
    if not all(isinstance(feature, PixelDifferenceFeature) for feature in (*root_features, *child_features)):
        raise TypeError("candidate pools must contain PixelDifferenceFeature instances")

    all_samples = np.ones(len(windows), dtype=bool)
    child_values = {feature: feature.evaluate(windows) for feature in child_features}
    best: TreeSearchResult | None = None
    for root_feature in root_features:
        root_values = root_feature.evaluate(windows)
        root_threshold, _ = _best_split(root_values, labels, weights, all_samples)
        root_left = root_values <= root_threshold
        child_nodes: list[TreeNode] = []
        leaf_masks: list[np.ndarray] = []
        total_error = 0.0
        for branch in (root_left, ~root_left):
            best_child: TreeNode | None = None
            best_child_error = float("inf")
            best_child_left: np.ndarray | None = None
            for feature, values in child_values.items():
                threshold, error = _best_split(values, labels, weights, branch)
                if error < best_child_error:
                    best_child = TreeNode(feature, threshold)
                    best_child_error = error
                    best_child_left = branch & (values <= threshold)
            assert best_child is not None and best_child_left is not None
            child_nodes.append(best_child)
            leaf_masks.extend((best_child_left, branch & ~best_child_left))
            total_error += best_child_error
        if best is None or total_error < best.weighted_error:
            scores = []
            for mask in leaf_masks:
                positive = float(weights[mask & (labels == 1)].sum())
                negative = float(weights[mask & (labels == 0)].sum())
                scores.append(_leaf_score(positive, negative))
            tree = Depth2WeakTree(
                TreeNode(root_feature, root_threshold),
                child_nodes[0],
                child_nodes[1],
                tuple(scores),
            )
            best = TreeSearchResult(tree, float(total_error))
    assert best is not None
    return best


def fit_random_depth2_tree(
    windows: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
    root_candidates: int = 32,
    child_candidates: int = 32,
) -> TreeSearchResult:
    """Train one tree with reproducibly sampled root and child candidates."""
    rng = np.random.default_rng(seed)
    roots = sample_random_features(rng, root_candidates)
    children = sample_random_features(rng, child_candidates)
    return search_depth2_tree(windows, labels, sample_weight, roots, children)
