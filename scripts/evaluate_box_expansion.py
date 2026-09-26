#!/usr/bin/env python3
"""Validation-only ablation of square proposal box geometry after verification."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.candidate_verifier import features
from src.detection_metrics import match_detections
from src.multiscale import PyramidConfig, detect_multiscale
from compare_multiscale import choose_pages


def expand(boxes: np.ndarray, width: int, height: int, fx: float, fy: float) -> np.ndarray:
    boxes = boxes.astype(np.float64)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) * np.array([fx, fy]) / 2
    result = np.rint(np.concatenate((center - half, center + half), axis=1)).astype(np.int32)
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height)
    return result


def main() -> None:
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = ROOT / "models/step4_cascade.json"
    linear_path = ROOT / "models/candidate_hog_svm.npz"
    rbf_path = ROOT / "models/candidate_hog_rbf_step3_v1.joblib"
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    with np.load(linear_path) as data:
        linear_coef, linear_intercept = data["coefficients"], float(data["intercept"])
    rbf = joblib.load(rbf_path)
    variants = [(1.0, 1.0), (1.25, 1.25), (1.5, 1.5), (1.75, 1.75),
                (2.0, 2.0), (1.25, 1.75), (1.5, 2.0), (1.75, 2.25)]
    rules = {"linear0.75": ("linear", 0.75), "rbf1.25": ("rbf", 1.25), "rbf1.5": ("rbf", 1.5)}
    totals = {rule: {f"{fx:g}x{fy:g}": {"tp": 0, "fp": 0, "fn": 0} for fx, fy in variants} for rule in rules}
    pages = []
    for page in choose_pages(manifest, 8):
        start = perf_counter()
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(page["image_path"])
        gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        boxes, scores, layers = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
        x = features(gray, boxes)
        verification = {"linear": x @ linear_coef + linear_intercept, "rbf": rbf.decision_function(x)}
        item = {"image_id": page["image_id"], "source_group": page["source_group"], "gt": len(gt),
                "nms_candidates": len(boxes), "layers": layers, "rules": {}}
        for rule, (model, threshold) in rules.items():
            keep = (verification[model] >= threshold) & (boxes[:, 2] - boxes[:, 0] >= 36)
            selected_boxes, selected_scores = boxes[keep], scores[keep]
            item["rules"][rule] = {}
            for fx, fy in variants:
                label = f"{fx:g}x{fy:g}"
                adjusted = expand(selected_boxes, gray.shape[1], gray.shape[0], fx, fy)
                result = match_detections(adjusted, selected_scores, gt, 0.5)
                item["rules"][rule][label] = {key: result[key] for key in ("tp", "fp", "fn")}
                for key in ("tp", "fp", "fn"):
                    totals[rule][label][key] += result[key]
        item["seconds"] = perf_counter() - start
        pages.append(item)
        print(page["image_id"], round(item["seconds"], 2), flush=True)
    for entries in totals.values():
        for result in entries.values():
            tp, fp, fn = result["tp"], result["fp"], result["fn"]
            result["precision"] = tp / (tp + fp) if tp + fp else 0
            result["recall"] = tp / (tp + fn) if tp + fn else 0
            result["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    output = {"split": "validation", "page_selection": "step5-comparison SHA-256 order, eight distinct source groups",
              "iou_threshold": 0.5, "source_box_convention": "xyxy half-open; expansion about box center and clipped to image",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (manifest, cascade_path, linear_path, rbf_path)},
              "pages": pages, "summary": totals, "mean_seconds": sum(p["seconds"] for p in pages) / len(pages)}
    target = ROOT / "results/step8_box_expansion_validation.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(totals)


if __name__ == "__main__":
    main()
