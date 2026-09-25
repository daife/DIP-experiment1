"""Four-stage landmark regression with local binary fern codes."""

from pathlib import Path

import numpy as np

from .landmark_regression import features


def fern_features(images: np.ndarray, shapes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Two four-bit fern leaves per landmark, encoded as 32 one-hot columns."""
    differences = features(images, shapes, offsets).reshape(len(images), 28, 8)
    bits = (differences > 0).astype(np.int32)
    codes = (bits.reshape(len(images), 28, 2, 4) * (1 << np.arange(4))).sum(axis=-1)
    result = np.zeros((len(images), 28, 32), dtype=np.float64)
    rows = np.arange(len(images))[:, None, None]
    points = np.arange(28)[None, :, None]
    ferns = np.arange(2)[None, None, :]
    result[rows, points, ferns * 16 + codes] = 1
    return result


class FernLandmarkRegressor:
    def __init__(self, mean: np.ndarray, offsets: np.ndarray, coefficients: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def predict_normalized(self, images: np.ndarray) -> np.ndarray:
        shape = np.broadcast_to(self.mean, (len(images), 28, 2)).copy()
        for coef in self.coefficients:
            x = fern_features(images, shape, self.offsets)
            shape += np.einsum("npf,pfc->npc", x, coef)
            shape = np.clip(shape, -0.25, 1.25)
        return shape

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, offsets=self.offsets, coefficients=self.coefficients)

    @classmethod
    def load(cls, path: Path) -> "FernLandmarkRegressor":
        with np.load(path) as data:
            return cls(data["mean"], data["offsets"], data["coefficients"])


def train_fern(images: np.ndarray, targets: np.ndarray, mask: np.ndarray, mean: np.ndarray,
               offsets: np.ndarray, stages: int = 4, alpha: float = 10.0) -> FernLandmarkRegressor:
    shape = np.broadcast_to(mean, targets.shape).copy()
    stages_coef = []
    for _ in range(stages):
        x = fern_features(images, shape, offsets)
        residual = targets - shape
        coef = np.zeros((28, 32, 2), dtype=np.float64)
        for point in range(28):
            valid = mask[:, point]
            if valid.sum() < 2:
                raise ValueError(f"Too few valid training points for landmark {point}")
            local = x[valid, point]
            coef[point] = np.linalg.solve(local.T @ local + alpha * np.eye(32),
                                          local.T @ residual[valid, point])
        shape += np.einsum("npf,pfc->npc", x, coef)
        shape = np.clip(shape, -0.25, 1.25)
        stages_coef.append(coef)
    return FernLandmarkRegressor(mean, offsets, np.stack(stages_coef))
