#!/usr/bin/env python3
"""Train a HOG candidate verifier from train Manga109 pages only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.candidate_verifier import features
from src.multiscale import PyramidConfig, detect_multiscale


def iou_max(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(boxes) or not len(gt):
        return np.zeros(len(boxes))
    tl = np.maximum(boxes[:, None, :2], gt[None, :, :2])
    br = np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    wh = np.maximum(0, br - tl)
    inter = wh[:, :, 0] * wh[:, :, 1]
    ba = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    ga = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    return (inter / (ba[:, None] + ga[None, :] - inter)).max(axis=1)


def choose_pages(manifest: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == "train" and row["annotations"]]
    rows.sort(key=lambda row: hashlib.sha256(("step8-verifier:" + row["image_id"]).encode()).digest())
    chosen, groups = [], set()
    for row in rows:
        if row["source_group"] not in groups:
            chosen.append(row)
            groups.add(row["source_group"])
            if len(chosen) == count:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--extra-positive-groups", type=int, default=0)
    parser.add_argument("--anime-positives", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "models/candidate_hog_svm.npz")
    args = parser.parse_args()
    if args.pages < 1 or args.step < 1:
        parser.error("pages and step must be positive")
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    model_path = ROOT / "models/step4_cascade.json"
    cascade = cascade_from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    rng = np.random.default_rng(20260925)
    xs, ys, summary = [], [], []
    for page in choose_pages(manifest, args.pages):
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(page["image_path"])
        gt = np.asarray([a["bbox"] for a in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        boxes, scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, args.step, 0.3))
        overlaps = iou_max(boxes, gt)
        positive = np.flatnonzero(overlaps >= 0.5)
        negative = np.flatnonzero(overlaps < 0.1)
        hard = negative[np.argsort(-scores[negative], kind="stable")[:60]]
        rest = np.setdiff1d(negative, hard)
        random = rng.choice(rest, size=min(60, len(rest)), replace=False)
        chosen = np.concatenate((positive, hard, random))
        if len(chosen):
            xs.append(features(gray, boxes[chosen]))
            ys.append((overlaps[chosen] >= 0.5).astype(np.uint8))
        # Add annotated faces as positives to cover missed sizes and poses.
        clipped = gt.copy()
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, gray.shape[1])
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, gray.shape[0])
        clipped = clipped[np.all(clipped[:, 2:] > clipped[:, :2], axis=1)]
        xs.append(features(gray, clipped))
        ys.append(np.ones(len(clipped), dtype=np.uint8))
        summary.append({"id": page["image_id"], "gt": len(gt), "proposals": len(boxes), "positive_proposals": len(positive), "sampled_negative": len(hard) + len(random)})
        print(summary[-1], flush=True)
    extra_summary = []
    if args.extra_positive_groups:
        scanned = {row["id"] for row in summary}
        for page in choose_pages(manifest, args.extra_positive_groups):
            if page["image_id"] in scanned:
                continue
            gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
            gt = np.asarray([a["bbox"] for a in page["annotations"]], dtype=np.int32).reshape(-1, 4)
            gt[:, [0, 2]] = np.clip(gt[:, [0, 2]], 0, gray.shape[1])
            gt[:, [1, 3]] = np.clip(gt[:, [1, 3]], 0, gray.shape[0])
            gt = gt[np.all(gt[:, 2:] > gt[:, :2], axis=1)]
            chosen = rng.choice(len(gt), size=min(10, len(gt)), replace=False)
            xs.append(features(gray, gt[chosen]))
            ys.append(np.ones(len(chosen), dtype=np.uint8))
            extra_summary.append({"id": page["image_id"], "positives": len(chosen)})
    anime_count = 0
    if args.anime_positives:
        review_set = ROOT / "datasets/annotations/review_sets/landmark320/review_set.json"
        review = json.loads(review_set.read_text(encoding="utf-8"))
        for row in review["images"]:
            if row["split"] != "train":
                continue
            gray = cv2.imread(str(ROOT / "datasets" / row["image_path"]), cv2.IMREAD_GRAYSCALE)
            xs.append(features(gray, np.array([[0, 0, gray.shape[1], gray.shape[0]]], dtype=np.int32)))
            ys.append(np.ones(1, dtype=np.uint8))
            anime_count += 1
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = LinearSVC(C=0.1, class_weight="balanced", max_iter=10000, random_state=20260925).fit(x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coefficients=model.coef_[0], intercept=model.intercept_[0])
    report = {"split": "train", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "cascade_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "seed": 20260925,
              "pages": summary, "extra_positive_pages": extra_summary, "anime_train_positives": anime_count,
              "positive_samples": int(y.sum()), "negative_samples": int((1-y).sum()),
              "hog": {"window": 24, "block": 12, "stride": 6, "cell": 6, "bins": 9}, "svm_C": 0.1}
    args.output.with_suffix(".json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print({"positive": report["positive_samples"], "negative": report["negative_samples"], "model": str(args.output)})


if __name__ == "__main__":
    main()
