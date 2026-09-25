"""HOG-based verification of Cascade candidate boxes."""

from pathlib import Path

import cv2
import numpy as np


def hog(crop: np.ndarray) -> np.ndarray:
    image = crop.astype(np.float32)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1] = image[2:] - image[:-2]
    magnitude = np.hypot(gx, gy)
    bins = np.floor(np.mod(np.arctan2(gy, gx), np.pi) * (9 / np.pi)).astype(np.intp)
    cells = np.zeros((4, 4, 9), dtype=np.float32)
    for cy in range(4):
        for cx in range(4):
            ys = slice(cy * 6, (cy + 1) * 6)
            xs = slice(cx * 6, (cx + 1) * 6)
            cells[cy, cx] = np.bincount(bins[ys, xs].ravel(), weights=magnitude[ys, xs].ravel(), minlength=9)[:9]
    blocks = []
    for cy in range(3):
        for cx in range(3):
            block = cells[cy:cy+2, cx:cx+2].ravel()
            blocks.append(block / np.sqrt(np.dot(block, block) + 1e-6))
    return np.concatenate(blocks)


def features(gray: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    result = []
    for x1, y1, x2, y2 in boxes:
        crop = cv2.resize(gray[y1:y2, x1:x2], (24, 24), interpolation=cv2.INTER_AREA)
        result.append(hog(crop))
    return np.asarray(result, dtype=np.float32).reshape(-1, 324)


class CandidateVerifier:
    def __init__(self, path: str | Path):
        with np.load(path) as data:
            self.coefficients = data["coefficients"]
            self.intercept = float(data["intercept"])
        if self.coefficients.shape != (324,):
            raise ValueError("invalid HOG verifier coefficients")

    def scores(self, gray: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        return features(gray, boxes) @ self.coefficients + self.intercept
