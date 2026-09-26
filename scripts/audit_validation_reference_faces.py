#!/usr/bin/env python3
"""Reference-only YOLO face proposals for checking Manga109 validation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from anime_face_detector import get_checkpoint_path
from anime_face_detector._face import load_face_detector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_multiscale import choose_pages


def overlaps(box: np.ndarray, gt: np.ndarray) -> float:
    if not len(gt):
        return 0.0
    intersection_size = np.maximum(0, np.minimum(box[2:4], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2]))
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    gt_area = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    return float(np.max(intersection / (box_area + gt_area - intersection)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/derived/step8_validation_reference_audit")
    args = parser.parse_args()
    if args.pages < 1 or not 0 <= args.threshold <= 1:
        parser.error("invalid page count or confidence threshold")
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    checkpoint = get_checkpoint_path("yolov3")
    model = load_face_detector("yolov3", str(checkpoint), device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_number, page in enumerate(choose_pages(manifest, args.pages)):
        image = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(page["image_path"])
        gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.float64).reshape(-1, 4)
        start = perf_counter()
        detections = model.detect(image)
        elapsed = perf_counter() - start
        kept = [box for box in detections if float(box[4]) >= args.threshold]
        canvas = image.copy()
        for i, box in enumerate(gt):
            x1, y1, x2, y2 = map(int, np.rint(box))
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), (255, 0, 0), 3)
            cv2.putText(canvas, f"G{i}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        records = []
        for i, box in enumerate(kept):
            x1, y1, x2, y2 = map(int, np.rint(box[:4]))
            iou = overlaps(box, gt)
            records.append({"id": f"R{i}", "bbox_xyxy": [x1, y1, x2, y2], "confidence": float(box[4]),
                            "max_original_gt_iou": iou, "review_status": "unreviewed"})
            color = (0, 0, 255) if iou < 0.1 else (0, 180, 0)
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), color, 2)
            cv2.putText(canvas, f"R{i}", (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        overlay_name = f"reference_{page_number:02d}_{page['source_group']}.png"
        cv2.imwrite(str(args.output / overlay_name), canvas)
        item = {"image_id": page["image_id"], "image_path": page["image_path"],
                "source_group": page["source_group"], "original_gt": len(gt), "reference_count": len(records),
                "reference_far_from_gt": sum(row["max_original_gt_iou"] < 0.1 for row in records),
                "seconds": elapsed, "overlay": overlay_name, "proposals": records}
        pages.append(item)
        print({k: item[k] for k in ("image_id", "original_gt", "reference_count", "reference_far_from_gt", "seconds")}, flush=True)
    output = {"split": "validation", "selection": "step5-comparison SHA-256 order, distinct source groups",
              "method": "anime-face-detector pretrained YOLOv3, CPU; reference audit only, not final detector or ground truth",
              "confidence_threshold": args.threshold, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "review_status": "all reference proposals unreviewed until visual check", "pages": pages,
              "summary": {"pages": len(pages), "original_gt": sum(p["original_gt"] for p in pages),
                          "reference_count": sum(p["reference_count"] for p in pages),
                          "reference_far_from_gt": sum(p["reference_far_from_gt"] for p in pages)}}
    (args.output / "reference_proposals.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
