#!/usr/bin/env python3
"""Render current detections and original annotations for a focused page audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.candidate_verifier import CandidateVerifier
from src.detection_metrics import match_detections
from src.multiscale import PyramidConfig, detect_multiscale
from compare_multiscale import choose_pages


def max_iou(box: np.ndarray, gt: np.ndarray) -> float:
    if not len(gt):
        return 0.0
    overlap = np.maximum(0, np.minimum(box[2:], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2]))
    intersection = overlap[:, 0] * overlap[:, 1]
    areas = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    box_area = np.prod(box[2:] - box[:2])
    return float(np.max(intersection / (box_area + areas - intersection)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/derived/step8_validation_label_audit")
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be positive")
    config_path = ROOT / "models/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cascade_path = ROOT / "models" / config["cascade"]
    verifier_path = ROOT / "models" / config["candidate_verifier"]["model"]
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    verifier = CandidateVerifier(verifier_path)
    search = config["search"]
    pyramid = PyramidConfig(search["scale_factor"], search["step"], search["nms_iou"])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    for page_number, page in enumerate(choose_pages(manifest, args.pages)):
        image = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(page["image_path"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes, scores, _ = detect_multiscale(gray, cascade, pyramid)
        verifier_scores = verifier.scores(gray, boxes)
        keep = (verifier_scores >= config["candidate_verifier"]["threshold"]) & (
            boxes[:, 2] - boxes[:, 0] >= config["candidate_verifier"]["min_side"])
        boxes, scores, verifier_scores = boxes[keep], scores[keep], verifier_scores[keep]
        gt = np.asarray([a["bbox"] for a in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        matched = match_detections(boxes, scores, gt, 0.5)
        matched_ids = {m["prediction_index"] for m in matched["matches"]}
        canvas = image.copy()
        for index, box in enumerate(gt):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), (255, 0, 0), 3)
            cv2.putText(canvas, f"G{index}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        for index, (box, score, verifier_score) in enumerate(zip(boxes, scores, verifier_scores)):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 180, 0) if index in matched_ids else (0, 0, 255)
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), color, 2)
            cv2.putText(canvas, f"P{index}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            rows.append({"page_id": page["image_id"], "parent_image_path": page["image_path"],
                         "prediction_id": f"P{index}", "bbox_xyxy": json.dumps(box.tolist()),
                         "cascade_score": float(score), "verifier_score": float(verifier_score),
                         "max_original_gt_iou": max_iou(box, gt), "matched_original_gt": int(index in matched_ids),
                         "visual_decision": "unreviewed", "review_note": ""})
        output_name = f"page_{page_number:02d}_{page['source_group']}.png"
        cv2.imwrite(str(args.output / output_name), canvas)
        summaries.append({"page_id": page["image_id"], "image_path": page["image_path"],
                          "overlay": output_name, "gt": len(gt), "predictions": len(boxes),
                          "original_matches": matched["tp"]})
        print(summaries[-1], flush=True)
    with (args.output / "prediction_review_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["page_id"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"split": "validation", "selection": "step5-comparison SHA-256 order",
              "colors": "blue=original GT, green=matched prediction, red=unmatched prediction",
              "review_status": "unreviewed; overlay is for human inspection, not evidence of corrected labels",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (manifest, config_path, cascade_path, verifier_path)},
              "pages": summaries}
    (args.output / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
