"""One-to-one, score-ordered page detection matching."""

from __future__ import annotations

import numpy as np


def match_detections(boxes: np.ndarray, scores: np.ndarray, ground_truth: np.ndarray,
                     threshold: float = 0.5) -> dict:
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    ground_truth = np.asarray(ground_truth).reshape(-1, 4)
    if boxes.shape != (len(scores), 4) or ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
        raise ValueError("invalid detection or ground-truth shape")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    used = np.zeros(len(ground_truth), dtype=bool)
    matches = []
    for index in np.argsort(-scores, kind="stable"):
        if not len(ground_truth):
            break
        box = boxes[index]
        overlap = np.maximum(0, np.minimum(box[2:], ground_truth[:, 2:]) -
                             np.maximum(box[:2], ground_truth[:, :2]))
        intersection = overlap[:, 0] * overlap[:, 1]
        box_area = np.prod(box[2:] - box[:2])
        gt_area = np.prod(ground_truth[:, 2:] - ground_truth[:, :2], axis=1)
        iou = intersection / (box_area + gt_area - intersection)
        iou[used] = -1
        best = int(np.argmax(iou))
        if iou[best] >= threshold:
            used[best] = True
            matches.append({"prediction_index": int(index), "ground_truth_index": best,
                            "iou": float(iou[best])})
    tp = len(matches)
    fp = len(boxes) - tp
    fn = len(ground_truth) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "matches": matches,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
