#!/usr/bin/env python3
"""Evaluate learned proposal box refinement on fixed validation pages."""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import CandidateVerifier, features
from src.cascade import cascade_from_dict
from src.detection_metrics import match_detections
from src.multiscale import PyramidConfig, detect_multiscale, nms
from compare_multiscale import choose_pages


def transform(boxes: np.ndarray, predictions: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = boxes.astype(np.float64)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    delta = np.clip(predictions, [-1.5, -1.5, -1.0, -1.0], [1.5, 1.5, 1.0, 1.0])
    new_center = centers + delta[:, :2] * sizes
    new_size = sizes * np.exp(delta[:, 2:])
    adjusted = np.rint(np.concatenate((new_center - new_size / 2, new_center + new_size / 2), axis=1)).astype(np.int32)
    adjusted[:, [0, 2]] = np.clip(adjusted[:, [0, 2]], 0, width)
    adjusted[:, [1, 3]] = np.clip(adjusted[:, [1, 3]], 0, height)
    valid = np.all(adjusted[:, 2:] > adjusted[:, :2], axis=1)
    return adjusted, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nms-iou', type=float, default=0.3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/step8_box_refiner_ablation_validation.json')
    args = parser.parse_args()
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = ROOT / "models/step4_cascade.json"
    verifier_path = ROOT / "models/candidate_hog_rbf_step3_v1.joblib"
    refiner_path = ROOT / "models/box_refiner_ridge_v1.joblib"
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    verifier = CandidateVerifier(verifier_path)
    refiner = joblib.load(refiner_path)
    thresholds = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    variants = ["current_rbf1.5", "unrefined_rbf_rank1.5", "unrefined_final_nms1.5", "refined_no_final_nms1.5"] + [f"refined_rbf{threshold:g}" for threshold in thresholds]
    totals = {name: {key: 0 for key in ("tp", "fp", "fn")} for name in variants}
    pages = []
    for page in choose_pages(manifest, 8):
        start = perf_counter()
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        boxes, cascade_scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, args.nms_iou))
        x = features(gray, boxes)
        rbf_scores = verifier.model.decision_function(x)
        predictions = refiner.predict(x)
        adjusted, valid = transform(boxes, predictions, gray.shape[1], gray.shape[0])
        item = {"image_id": page["image_id"], "gt": len(gt), "pre_nms_count": len(boxes), "results": {}}
        keep = (rbf_scores >= 1.5) & ((boxes[:, 2] - boxes[:, 0]) >= 36)
        current = match_detections(boxes[keep], cascade_scores[keep], gt, 0.5)
        item["results"]["current_rbf1.5"] = {k: current[k] for k in ("tp", "fp", "fn")}
        for key in ("tp", "fp", "fn"):
            totals["current_rbf1.5"][key] += current[key]
        # Separate box geometry from changes in score ranking and duplicate removal.
        ablations = {
            "unrefined_rbf_rank1.5": (boxes[keep], rbf_scores[keep], False),
            "unrefined_final_nms1.5": (boxes[keep], rbf_scores[keep], True),
            "refined_no_final_nms1.5": (adjusted[keep & valid], rbf_scores[keep & valid], False),
        }
        for name, (ablation_boxes, ablation_scores, merge) in ablations.items():
            indices = nms(ablation_boxes, ablation_scores, 0.3) if merge and len(ablation_boxes) else np.arange(len(ablation_boxes))
            result = match_detections(ablation_boxes[indices], ablation_scores[indices], gt, 0.5)
            item["results"][name] = {k: result[k] for k in ("tp", "fp", "fn")}
            for key in ("tp", "fp", "fn"):
                totals[name][key] += result[key]
        for threshold in thresholds:
            selected = (rbf_scores >= threshold) & ((boxes[:, 2] - boxes[:, 0]) >= 36) & valid
            selected_boxes = adjusted[selected]
            selected_scores = rbf_scores[selected]
            indices = nms(selected_boxes, selected_scores, 0.3) if len(selected_boxes) else np.empty(0, dtype=np.int64)
            result = match_detections(selected_boxes[indices], selected_scores[indices], gt, 0.5)
            name = f"refined_rbf{threshold:g}"
            item["results"][name] = {k: result[k] for k in ("tp", "fp", "fn")}
            for key in ("tp", "fp", "fn"):
                totals[name][key] += result[key]
        item["seconds"] = perf_counter() - start
        pages.append(item)
        print(page["image_id"], round(item["seconds"], 2), item["results"], flush=True)
    for result in totals.values():
        tp, fp, fn = (result[key] for key in ("tp", "fp", "fn"))
        result["precision"] = tp / (tp + fp) if tp + fp else 0
        result["recall"] = tp / (tp + fn) if tp + fn else 0
        result["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    output = {"split": "validation", "selection": "step5-comparison SHA-256 order, eight distinct source groups",
              "first_nms_iou": args.nms_iou, "final_nms_iou": 0.3, "step": 2, "scale_factor": 1.2,
              "iou_threshold": 0.5, "refinement": "Ridge HOG predicts dx,dy,dlogw,dlogh; deltas clipped; clipped image bounds; final NMS IoU 0.3 ranked by RBF score",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (manifest, cascade_path, verifier_path, refiner_path)},
              "timing_scope": "shared Cascade, HOG, Ridge and all ablation/threshold variants; not isolated deployed inference latency",
              "pages": pages, "summary": totals,
              "mean_seconds": sum(page["seconds"] for page in pages) / len(pages)}
    target = args.output
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(totals)


if __name__ == "__main__":
    main()
