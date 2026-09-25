"""Public end-to-end anime face detector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .cascade import cascade_from_dict
from .candidate_verifier import CandidateVerifier
from .landmark_regression import LandmarkRegressor
from .multiscale import PyramidConfig, detect_multiscale


class AnimeFaceDetector:
    def __init__(self, model_path: str | Path):
        root = Path(model_path)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != "experiment1-demo-v1":
            raise ValueError("unsupported detector config")
        self.config = config
        self.cascade = cascade_from_dict(json.loads((root / config["cascade"]).read_text(encoding="utf-8")))
        self.regressor = LandmarkRegressor.load(root / config["landmark_model"])
        verifier = config.get("candidate_verifier")
        self.verifier = CandidateVerifier(root / verifier["model"]) if verifier else None
        self.verifier_threshold = verifier["threshold"] if verifier else None
        self.verifier_min_side = verifier["min_side"] if verifier else None
        search = config["search"]
        self.pyramid = PyramidConfig(search["scale_factor"], search["step"], search["nms_iou"])

    def detect(self, image: np.ndarray) -> list[dict]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("image must be a nonempty OpenCV uint8 BGR image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes, scores, _ = detect_multiscale(gray, self.cascade, self.pyramid)
        if self.verifier is not None and len(boxes):
            verifier_scores = self.verifier.scores(gray, boxes)
            keep = (verifier_scores >= self.verifier_threshold) & (boxes[:, 2] - boxes[:, 0] >= self.verifier_min_side)
            boxes, scores = boxes[keep], scores[keep]
        return [
            {"bbox": box.tolist(), "score": float(score),
             "landmarks": self.regressor.predict_image(image, box).tolist()}
            for box, score in zip(boxes, scores)
        ]
