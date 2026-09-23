"""Multi-scale 24x24 Cascade search and score-ordered box suppression."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .cascade import Cascade
from .page_scan import scan_page


@dataclass(frozen=True)
class PyramidConfig:
    scale_factor: float = 1.2
    step: int = 1
    nms_iou: float = 0.3

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale_factor) or self.scale_factor <= 1:
            raise ValueError("scale_factor must be finite and greater than 1")
        if type(self.step) is not int or self.step < 1:
            raise ValueError("step must be a positive integer")
        if not np.isfinite(self.nms_iou) or not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou must be in [0, 1]")


def pyramid_sizes(width: int, height: int, scale_factor: float) -> list[tuple[int, int]]:
    """Unique (width, height) sizes, stopping before either side falls below 24."""
    if min(width, height) < 1 or not np.isfinite(scale_factor) or scale_factor <= 1:
        raise ValueError("positive image dimensions and scale_factor > 1 required")
    sizes = []
    level = 0
    while True:
        size = (round(width / scale_factor**level), round(height / scale_factor**level))
        if min(size) < 24:
            break
        if not sizes or size != sizes[-1]:
            sizes.append(size)
        level += 1
    return sizes


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """Return kept indices in descending score order; boxes use exclusive x2/y2."""
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),):
        raise ValueError("boxes must be (N,4) and scores must be (N,)")
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        raise ValueError("boxes and scores must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must have positive area")
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]).astype(np.float64) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    while len(order):
        current = order[0]
        kept.append(current)
        others = order[1:]
        intersection_width = np.maximum(0, np.minimum(boxes[current, 2], boxes[others, 2]) - np.maximum(boxes[current, 0], boxes[others, 0]))
        intersection_height = np.maximum(0, np.minimum(boxes[current, 3], boxes[others, 3]) - np.maximum(boxes[current, 1], boxes[others, 1]))
        intersection = intersection_width.astype(np.float64) * intersection_height
        iou = intersection / (areas[current] + areas[others] - intersection)
        order = others[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def detect_multiscale(gray: np.ndarray, cascade: Cascade, config: PyramidConfig = PyramidConfig()) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return original-image boxes, last-stage scores, and per-level timings.

    Each layer is resized directly from the input. Coordinates are rounded to
    the original pixel grid and clipped; x2/y2 remain exclusive.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("gray must be a nonempty uint8 grayscale image")
    height, width = gray.shape
    all_boxes, all_scores, layers = [], [], []
    for level, (layer_width, layer_height) in enumerate(pyramid_sizes(width, height, config.scale_factor)):
        start = perf_counter()
        layer = gray if (layer_width, layer_height) == (width, height) else cv2.resize(gray, (layer_width, layer_height), interpolation=cv2.INTER_AREA)
        boxes, scores = scan_page(layer, cascade, step=config.step)
        elapsed = perf_counter() - start
        windows = ((layer_width - 24) // config.step + 1) * ((layer_height - 24) // config.step + 1)
        layers.append({"level": level, "width": layer_width, "height": layer_height,
                       "windows": windows, "candidates": len(boxes), "seconds": elapsed})
        if len(boxes):
            mapped = np.rint(boxes.astype(np.float64) * np.array([width / layer_width, height / layer_height] * 2)).astype(np.int32)
            mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, width)
            mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, height)
            all_boxes.append(mapped)
            all_scores.append(scores)
    if not all_boxes:
        return np.empty((0, 4), dtype=np.int32), np.empty(0, dtype=np.float64), layers
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    kept = nms(boxes, scores, config.nms_iou)
    return boxes[kept], scores[kept], layers
