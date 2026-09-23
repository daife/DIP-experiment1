"""Real AdaBoost stages built from depth-2 pixel-difference trees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .weak_tree import Depth2WeakTree, PixelDifferenceFeature, TreeNode, _check_windows
from .weak_tree_training import fit_random_depth2_tree


@dataclass(frozen=True, slots=True)
class CascadeStage:
    trees: tuple[Depth2WeakTree, ...]
    threshold: float
    learning_rate: float = 1.0

    def scores(self, windows: np.ndarray) -> np.ndarray:
        _check_windows(windows)
        if windows.ndim != 4:
            raise ValueError("stage scoring requires a batch")
        scores = np.zeros(len(windows), dtype=np.float64)
        for tree in self.trees:
            scores += self.learning_rate * tree.predict_scores(windows)
        return scores


@dataclass(frozen=True, slots=True)
class Cascade:
    stages: tuple[CascadeStage, ...]

    def evaluate(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Return pass mask, final-stage scores, and counts entering/leaving each stage."""
        _check_windows(windows)
        if windows.ndim != 4:
            raise ValueError("cascade evaluation requires a batch")
        active = np.ones(len(windows), dtype=bool)
        scores = np.full(len(windows), -np.inf, dtype=np.float64)
        trace = []
        for stage in self.stages:
            indices = np.flatnonzero(active)
            if len(indices):
                scores[indices] = stage.scores(windows[indices])
                active[indices] = scores[indices] >= stage.threshold
            trace.append({"entering": int(len(indices)), "passing": int(active.sum())})
        return active, scores, trace


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    positives = int((labels == 1).sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("stage needs both positive and negative training samples")
    return np.where(labels == 1, 0.5 / positives, 0.5 / negatives)


def threshold_for_recall(scores: np.ndarray, target_recall: float) -> float:
    """Largest threshold retaining at least the target fraction of positives."""
    if not 0 < target_recall <= 1 or len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("need finite positive scores and target_recall in (0, 1]")
    rejected = int(np.floor((1 - target_recall) * len(scores) + 1e-12))
    return float(np.sort(scores)[rejected])


def fit_stage(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_positive_windows: np.ndarray,
    *,
    seed: int,
    num_trees: int = 6,
    root_candidates: int = 16,
    child_candidates: int = 16,
    target_recall: float = 0.99,
    learning_rate: float = 0.5,
) -> CascadeStage:
    """Fit one stage and calibrate its threshold on validation positives."""
    _check_windows(train_windows)
    _check_windows(validation_positive_windows)
    if train_windows.ndim != 4 or validation_positive_windows.ndim != 4:
        raise ValueError("training and validation windows must be batches")
    labels = np.asarray(train_labels)
    if labels.shape != (len(train_windows),) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("train_labels must contain one binary label per window")
    if num_trees < 1 or not 0 < learning_rate <= 1:
        raise ValueError("num_trees must be positive and learning_rate in (0, 1]")
    weights = _balanced_weights(labels)
    signed_labels = np.where(labels == 1, 1, -1)
    trees = []
    for index in range(num_trees):
        result = fit_random_depth2_tree(
            train_windows, labels, weights,
            seed=seed + index, root_candidates=root_candidates,
            child_candidates=child_candidates,
        )
        tree = result.tree
        trees.append(tree)
        margin = learning_rate * signed_labels * tree.predict_scores(train_windows)
        weights *= np.exp(-np.clip(margin, -50, 50))
        weights /= weights.sum()
    provisional = CascadeStage(tuple(trees), 0.0, learning_rate)
    threshold = threshold_for_recall(provisional.scores(validation_positive_windows), target_recall)
    return CascadeStage(tuple(trees), threshold, learning_rate)


def fit_cascade(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    validation_windows: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int = 20260923,
    num_stages: int = 3,
    num_trees: int = 6,
    root_candidates: int = 16,
    child_candidates: int = 16,
    target_recall: float = 0.99,
    learning_rate: float = 0.5,
) -> Cascade:
    """Train stages on positives and negatives surviving preceding stages."""
    _check_windows(train_windows)
    _check_windows(validation_windows)
    train_labels = np.asarray(train_labels)
    validation_labels = np.asarray(validation_labels)
    if train_windows.ndim != 4 or validation_windows.ndim != 4:
        raise ValueError("windows must be batches")
    if train_labels.shape != (len(train_windows),) or validation_labels.shape != (len(validation_windows),):
        raise ValueError("label lengths must match window counts")
    if not np.all(np.isin(train_labels, (0, 1))) or not np.all(np.isin(validation_labels, (0, 1))):
        raise ValueError("labels must be binary")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    train_alive = np.ones(len(train_windows), dtype=bool)
    validation_alive = np.ones(len(validation_windows), dtype=bool)
    stages = []
    for stage_index in range(num_stages):
        # Keep all reviewed positives in stage fitting, including those lost by
        # earlier stages; only surviving negatives are passed to later stages.
        selected_train = (train_labels == 1) | ((train_labels == 0) & train_alive)
        selected_validation = (validation_labels == 1) & validation_alive
        if not np.any((train_labels == 0) & train_alive):
            raise ValueError(f"no training negatives survive before stage {stage_index}")
        if not np.any(selected_validation):
            raise ValueError(f"no validation positives survive before stage {stage_index}")
        stage = fit_stage(
            train_windows[selected_train], train_labels[selected_train],
            validation_windows[selected_validation],
            seed=seed + stage_index * 10000, num_trees=num_trees,
            root_candidates=root_candidates, child_candidates=child_candidates,
            target_recall=target_recall, learning_rate=learning_rate,
        )
        stages.append(stage)
        train_indices = np.flatnonzero(train_alive)
        validation_indices = np.flatnonzero(validation_alive)
        train_alive[train_indices] = stage.scores(train_windows[train_indices]) >= stage.threshold
        validation_alive[validation_indices] = stage.scores(validation_windows[validation_indices]) >= stage.threshold
    return Cascade(tuple(stages))


def stage_statistics(cascade: Cascade, windows: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Count face retention and background rejection at each stage."""
    labels = np.asarray(labels)
    if labels.shape != (len(windows),) or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must be binary and match windows")
    active = np.ones(len(windows), dtype=bool)
    report = []
    for index, stage in enumerate(cascade.stages):
        entering_positive = int(np.sum(active & (labels == 1)))
        entering_negative = int(np.sum(active & (labels == 0)))
        selected = np.flatnonzero(active)
        active[selected] = stage.scores(windows[selected]) >= stage.threshold
        passing_positive = int(np.sum(active & (labels == 1)))
        passing_negative = int(np.sum(active & (labels == 0)))
        report.append({
            "stage": index,
            "trees": len(stage.trees),
            "threshold": stage.threshold,
            "positive_entering": entering_positive,
            "positive_passing": passing_positive,
            "negative_entering": entering_negative,
            "negative_passing": passing_negative,
            "negative_rejected": entering_negative - passing_negative,
        })
    return report


def cascade_to_dict(cascade: Cascade) -> dict:
    def feature_dict(feature: PixelDifferenceFeature) -> dict:
        return {"channel": feature.channel, "p1": list(feature.p1), "p2": list(feature.p2)}

    def node_dict(node: TreeNode) -> dict:
        return {"feature": feature_dict(node.feature), "threshold": node.threshold}

    return {
        "format": "experiment1-depth2-cascade-v1",
        "window_size": 24,
        "channel_count": 11,
        "stages": [
            {"threshold": stage.threshold, "learning_rate": stage.learning_rate,
             "trees": [{"root": node_dict(tree.root), "left": node_dict(tree.left),
                        "right": node_dict(tree.right), "leaf_scores": list(tree.leaf_scores)}
                       for tree in stage.trees]}
            for stage in cascade.stages
        ],
    }


def cascade_from_dict(payload: dict) -> Cascade:
    if payload.get("format") != "experiment1-depth2-cascade-v1" or payload.get("window_size") != 24 or payload.get("channel_count") != 11:
        raise ValueError("unsupported cascade format")

    def node(data: dict) -> TreeNode:
        feature = data["feature"]
        return TreeNode(PixelDifferenceFeature(feature["channel"], tuple(feature["p1"]), tuple(feature["p2"])), data["threshold"])

    stages = []
    for data in payload["stages"]:
        trees = tuple(Depth2WeakTree(node(t["root"]), node(t["left"]), node(t["right"]), tuple(t["leaf_scores"])) for t in data["trees"])
        stages.append(CascadeStage(trees, data["threshold"], data["learning_rate"]))
    return Cascade(tuple(stages))
