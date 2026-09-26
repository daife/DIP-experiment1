#!/usr/bin/env python3
"""Fit a small HOG box regressor from Cascade proposals on train pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import features
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig, detect_multiscale
from train_candidate_verifier import choose_pages


def ious(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    tl = np.maximum(boxes[:, None, :2], gt[None, :, :2])
    br = np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    wh = np.maximum(0, br - tl)
    inter = wh[:, :, 0] * wh[:, :, 1]
    ba = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    ga = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    return inter / (ba[:, None] + ga[None, :] - inter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--per-face", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--model", type=Path, default=ROOT / "models/box_refiner_ridge_v1.joblib")
    parser.add_argument("--samples-output", type=Path, help="Save train HOG/targets and proposal provenance for controlled regression comparisons")
    args = parser.parse_args()
    if args.pages < 1 or not 0 < args.min_iou < 1 or args.per_face < 1 or args.alpha < 0:
        parser.error("invalid training parameters")
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = ROOT / "models/step4_cascade.json"
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    x_all, y_all, pages, provenance = [], [], [], []
    for page in choose_pages(manifest, args.pages):
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(page["image_path"])
        boxes, scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
        gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.float64).reshape(-1, 4)
        overlap = ious(boxes.astype(np.float64), gt)
        selected = []
        for face_index in range(len(gt)):
            eligible = np.flatnonzero(overlap[:, face_index] >= args.min_iou)
            eligible = sorted(eligible, key=lambda i: (-overlap[i, face_index], -scores[i]))[:args.per_face]
            for candidate_index in eligible:
                if candidate_index not in selected:
                    selected.append(int(candidate_index))
        if selected:
            candidate_boxes = boxes[selected].astype(np.float64)
            gt_indices = np.argmax(overlap[selected], axis=1)
            targets = []
            for box, gi in zip(candidate_boxes, gt_indices, strict=True):
                target = gt[gi]
                bw, bh = box[2] - box[0], box[3] - box[1]
                tw, th = target[2] - target[0], target[3] - target[1]
                bc = (box[:2] + box[2:]) / 2
                tc = (target[:2] + target[2:]) / 2
                targets.append([(tc[0] - bc[0]) / bw, (tc[1] - bc[1]) / bh,
                                np.log(tw / bw), np.log(th / bh)])
                provenance.append({"image_id": page["image_id"], "split": page["split"],
                                   "source_group": page["source_group"], "proposal": box.tolist(),
                                   "target": target.tolist(), "gt_index": int(gi)})
            x_all.append(features(gray, boxes[selected]))
            y_all.extend(targets)
        pages.append({"image_id": page["image_id"], "gt": len(gt), "nms_candidates": len(boxes),
                      "refiner_samples": len(selected)})
        print(pages[-1], flush=True)
    x, y = np.concatenate(x_all), np.asarray(y_all, dtype=np.float64)
    model = Ridge(alpha=args.alpha).fit(x, y)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model, compress=3)
    report = {"split": "train", "pages": pages, "page_count": len(pages), "samples": len(y),
              "min_iou": args.min_iou, "per_face": args.per_face, "alpha": args.alpha,
              "input": "324-dimensional 24x24 grayscale HOG; targets dx,dy,dlogw,dlogh normalized by proposal dimensions",
              "seed": "none; fixed SHA-256 page ordering and deterministic top-IoU selection",
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "cascade_sha256": hashlib.sha256(cascade_path.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()}
    args.model.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.samples_output:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.samples_output, x=x, y=y)
        report["samples_sha256"] = hashlib.sha256(args.samples_output.read_bytes()).hexdigest()
        report["provenance"] = provenance
        args.samples_output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({k: report[k] for k in ("samples", "min_iou", "per_face", "alpha", "model_sha256")})


if __name__ == "__main__":
    main()
