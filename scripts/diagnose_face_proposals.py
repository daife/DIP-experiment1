#!/usr/bin/env python3
"""Trace annotated faces through the fixed Cascade, NMS, and HOG gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.candidate_verifier import CandidateVerifier
from src.multiscale import PyramidConfig, detect_multiscale
from train_candidate_verifier import choose_pages as choose_train_pages
from compare_multiscale import choose_pages as choose_validation_pages


def overlaps(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    intersection_size = np.maximum(0, np.minimum(boxes[:, 2:], gt[2:]) - np.maximum(boxes[:, :2], gt[:2]))
    intersection = intersection_size[:, 0].astype(np.float64) * intersection_size[:, 1]
    box_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).astype(np.float64)
    gt_area = float(np.prod(gt[2:] - gt[:2]))
    return intersection / (box_area + gt_area - intersection)


def best(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray, levels: np.ndarray | None = None) -> dict | None:
    if not len(boxes):
        return None
    ious = overlaps(boxes, gt)
    index = int(np.argmax(ious))
    result = {"iou": float(ious[index]), "score": float(scores[index]), "bbox": boxes[index].tolist(), "index": index}
    if levels is not None:
        result["level"] = int(levels[index])
    return result


def diagnose(gt: np.ndarray, raw: dict, nms_boxes: np.ndarray, nms_scores: np.ndarray,
             hog_scores: np.ndarray, threshold: float, min_side: int) -> tuple[list[dict], dict]:
    raw_boxes, raw_scores, raw_levels = raw["boxes"], raw["scores"], raw["levels"]
    gate = (hog_scores >= threshold) & (nms_boxes[:, 2] - nms_boxes[:, 0] >= min_side)
    faces = []
    for index, face in enumerate(gt):
        before = best(raw_boxes, raw_scores, face, raw_levels)
        after_nms = best(nms_boxes, nms_scores, face)
        after_gate = best(nms_boxes[gate], nms_scores[gate], face)
        # Categories describe candidate availability at IoU >= .5, not one-to-one TP.
        if before is None:
            reason = "no_candidate"
        elif before["iou"] < 0.5:
            reason = "poor_localization"
        elif after_nms is None or after_nms["iou"] < 0.5:
            reason = "nms_suppressed"
        elif after_gate is None or after_gate["iou"] < 0.5:
            reason = "hog_or_side_filtered"
        else:
            reason = "candidate_retained"
        qualifying = np.flatnonzero(overlaps(nms_boxes, face) >= 0.5)
        hog_pass = int(np.sum(hog_scores[qualifying] >= threshold))
        side_pass = int(np.sum(nms_boxes[qualifying, 2] - nms_boxes[qualifying, 0] >= min_side))
        faces.append({"annotation_index": index, "gt_bbox": face.tolist(), "pre_nms_best": before,
                      "post_nms_best": after_nms, "post_filter_best": after_gate,
                      "post_nms_iou50_candidates": int(len(qualifying)), "hog_pass_count": hog_pass,
                      "side_pass_count": side_pass, "reason": reason})
    return faces, {"post_nms": int(len(nms_boxes)), "hog_pass": int(np.sum(hog_scores >= threshold)),
                   "side_pass": int(np.sum(nms_boxes[:, 2] - nms_boxes[:, 0] >= min_side)),
                   "final": int(np.sum(gate))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--output", type=Path)
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
    threshold = config["candidate_verifier"]["threshold"]
    min_side = config["candidate_verifier"]["min_side"]
    selected = (choose_train_pages if args.split == "train" else choose_validation_pages)(manifest, args.pages)
    pages = []
    for page in selected:
        image_path = ROOT / "datasets" / page["image_path"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        start = perf_counter()
        boxes, scores, layers, raw = detect_multiscale(gray, cascade, pyramid, return_pre_nms=True)
        hog = verifier.scores(gray, boxes)
        gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        faces, counts = diagnose(gt, raw, boxes, scores, hog, threshold, min_side)
        item = {"image_id": page["image_id"], "source_group": page["source_group"],
                "image_path": page["image_path"], "image_size": [gray.shape[1], gray.shape[0]],
                "ground_truth": len(gt), "layers": layers, "pre_nms_candidates": len(raw["boxes"]),
                "candidate_counts": counts, "faces": faces, "seconds": perf_counter() - start}
        pages.append(item)
        print({"id": page["image_id"], "faces": len(faces), "reasons": dict(Counter(f["reason"] for f in faces)),
               "pre_nms": len(raw["boxes"]), "seconds": round(item["seconds"], 2)}, flush=True)
    output = {"split": args.split, "selection": "step8-verifier SHA-256 order" if args.split == "train" else "step5-comparison SHA-256 order",
              "source_groups_distinct": True, "iou_threshold": 0.5, "config": {"search": search, "candidate_verifier": config["candidate_verifier"]},
              "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (manifest, config_path, cascade_path, verifier_path)},
              "pages": pages, "summary": {"pages": len(pages), "faces": sum(len(p["faces"]) for p in pages),
              "reasons": dict(Counter(f["reason"] for p in pages for f in p["faces"])),
              "seconds": sum(p["seconds"] for p in pages)}}
    target = args.output or ROOT / f"results/step8_proposal_diagnosis_{args.split}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
