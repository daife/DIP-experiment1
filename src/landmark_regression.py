"""Cascaded local pixel-difference Ridge landmark regressor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalized_image(gray: np.ndarray, bbox: np.ndarray, size: int = 256) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    x = np.linspace(x0, x1, size, dtype=np.float32)
    y = np.linspace(y0, y1, size, dtype=np.float32)
    mx, my = np.meshgrid(x, y)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0


def features(images: np.ndarray, shapes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Sample bilinear intensities at two offsets per feature around each current point."""
    n, size = len(images), images.shape[1]
    points = shapes[:, :, None, None, :] + offsets[None, :, :, :, :]
    xy = np.clip(points * (size - 1), 0, size - 1)
    x, y = xy[..., 0], xy[..., 1]
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, size - 1), np.minimum(y0 + 1, size - 1)
    dx, dy = x - x0, y - y0
    row = np.arange(n)[:, None, None, None]
    sampled = ((1 - dx) * (1 - dy) * images[row, y0, x0]
               + dx * (1 - dy) * images[row, y0, x1]
               + (1 - dx) * dy * images[row, y1, x0]
               + dx * dy * images[row, y1, x1])
    return (sampled[..., 0] - sampled[..., 1]).reshape(n, -1).astype(np.float64)


def fit_stage(x: np.ndarray, residual: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Fit independent masked Ridge targets; rows are [bias, coefficients]."""
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.zeros((design.shape[1], 56), dtype=np.float64)
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[0, 0] = 0
    for j in range(56):
        valid = mask[:, j // 2]
        if valid.sum() < 2:
            raise ValueError(f"Too few valid training points for output {j}")
        a = design[valid]
        coef[:, j] = np.linalg.solve(a.T @ a + regularizer, a.T @ residual[valid, j])
    return coef


class LandmarkRegressor:
    def __init__(self, mean: np.ndarray, offsets: np.ndarray, coefficients: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def predict_normalized(self, images: np.ndarray) -> np.ndarray:
        shape = np.broadcast_to(self.mean, (len(images), 28, 2)).copy()
        for coef in self.coefficients:
            x = features(images, shape, self.offsets)
            shape += (np.column_stack([np.ones(len(x)), x]) @ coef).reshape(-1, 28, 2)
            shape = np.clip(shape, -0.25, 1.25)
        return shape

    def predict_image(self, bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Return 28 XY points in original image coordinates for one face box."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        box = np.asarray(bbox, dtype=np.float64)
        shape = self.predict_normalized(normalized_image(gray, box)[None])[0]
        return shape * (box[2:] - box[:2]) + box[:2]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, offsets=self.offsets, coefficients=self.coefficients)

    @classmethod
    def load(cls, path: Path) -> "LandmarkRegressor":
        with np.load(path) as data:
            return cls(data["mean"], data["offsets"], data["coefficients"])


def train(images: np.ndarray, targets: np.ndarray, mask: np.ndarray, mean: np.ndarray,
          offsets: np.ndarray, stages: int = 4, alpha: float = 10.0) -> LandmarkRegressor:
    shape = np.broadcast_to(mean, targets.shape).copy()
    coefficients = []
    for _ in range(stages):
        x = features(images, shape, offsets)
        coef = fit_stage(x, (targets - shape).reshape(len(images), 56), mask, alpha)
        shape += (np.column_stack([np.ones(len(x)), x]) @ coef).reshape(-1, 28, 2)
        shape = np.clip(shape, -0.25, 1.25)
        coefficients.append(coef)
    return LandmarkRegressor(mean, offsets, np.stack(coefficients))


def make_offsets(seed: int, differences_per_point: int = 8, radius: float = 0.075) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-radius, radius, (28, differences_per_point, 2, 2))
