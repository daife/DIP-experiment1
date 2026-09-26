#!/usr/bin/env python3
"""Compare original and visually reviewed supplemental boxes on audited pages."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection_metrics import match_detections


def main() -> None:
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    detections_csv = ROOT / "datasets/derived/step8_validation_label_audit/prediction_review_template.csv"
    supplemental_csv = ROOT / "datasets/annotations/corrected/step8_validation_reference_face_review.csv"
    annotations = {row["image_id"]: row for row in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())}
    predictions: dict[str, list[dict]] = {}
    with detections_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            predictions.setdefault(row["page_id"], []).append(row)
    additions: dict[str, list[dict]] = {}
    with supplemental_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["review_decision"] != "visible_face_missing_from_original_gt":
                raise ValueError("supplemental annotations must be visually confirmed")
            additions.setdefault(row["page_id"], []).append(row)
    pages = []
    totals = {key: {name: 0 for name in ("tp", "fp", "fn")} for key in ("original_gt", "supplemented_gt")}
    for page_id, rows in predictions.items():
        page = annotations[page_id]
        boxes = np.asarray([json.loads(row["bbox_xyxy"]) for row in rows], dtype=np.int32).reshape(-1, 4)
        scores = np.asarray([float(row["cascade_score"]) for row in rows], dtype=np.float64)
        original_gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        supplement = np.asarray([json.loads(row["bbox_xyxy"]) for row in additions.get(page_id, [])], dtype=np.int32).reshape(-1, 4)
        results = {}
        for name, gt in (("original_gt", original_gt), ("supplemented_gt", np.concatenate((original_gt, supplement)))):
            result = match_detections(boxes, scores, gt, 0.5)
            results[name] = {key: result[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}
            for key in ("tp", "fp", "fn"):
                totals[name][key] += result[key]
        pages.append({"page_id": page_id, "original_gt": len(original_gt), "supplemental_gt": len(supplement),
                      "predictions": len(boxes), "results": results})
    for result in totals.values():
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        result["precision"] = tp / (tp + fp) if tp + fp else 0
        result["recall"] = tp / (tp + fn) if tp + fn else 0
        result["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    output = {"split": "validation", "scope": "two visually audited pages only",
              "metric": "IoU>=0.5 one-to-one score ordered matching",
              "supplement_source": "10 human-confirmed face boxes proposed by pretrained YOLOv3; proposal boxes are coarse, not precision-edited ground truth",
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (manifest, detections_csv, supplemental_csv)},
              "pages": pages, "summary": totals}
    target = ROOT / "results/step8_validation_label_sensitivity_2pages.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
