#!/usr/bin/env python3
"""Train an RBF HOG verifier on reviewed train crops only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import hog
from src.candidate_verifier import features as candidate_features
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig, detect_multiscale
from train_candidate_verifier import choose_pages, iou_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    parser.add_argument("--model", type=Path, default=ROOT / "models/candidate_hog_rbf_step3_v1.joblib")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--proposal-positive-pages", type=int, default=0)
    parser.add_argument("--reviewed-negative-csv", type=Path)
    parser.add_argument("--positive-margin", type=float, help="Regenerate centered square train positive crops from reviewed source faces at this side/max-face-side ratio")
    parser.add_argument('--features-output', type=Path, help='Save train features/labels for comparisons on identical examples')
    args = parser.parse_args()
    summary = json.loads((args.dataset / "review_summary.json").read_text(encoding="utf-8"))
    if summary.get("fully_reviewed") is not True:
        raise ValueError("crop review must be complete")
    manifest = args.dataset / "usable_samples.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == "train" and row["review_decision"] == "keep"]
    x, y = [], []
    if args.positive_margin is not None and args.positive_margin < 1:
        parser.error('positive margin must be at least 1 to contain the reviewed face')
    page_image, page_path = None, None
    augmentation_provenance = []
    ordered_rows = sorted(rows, key=lambda r: (r['parent_image_path'], r['sample_id'])) if args.positive_margin is not None else rows
    for row in ordered_rows:
        if args.positive_margin is not None and row['label'] == 1:
            if row['parent_image_path'] != page_path:
                page_path = row['parent_image_path']
                page_image = cv2.imread(str(ROOT / 'datasets' / page_path), 0)
            if page_image is None:
                raise FileNotFoundError(page_path)
            box = np.asarray(row['source_bbox_xyxy'], dtype=float)
            center = (box[:2] + box[2:]) / 2
            side = max(box[2:] - box[:2]) * args.positive_margin
            lo = np.floor(center-side/2).astype(int)
            hi = lo + int(np.ceil(side))
            padding = (max(0,-lo[1]), max(0,hi[1]-page_image.shape[0]), max(0,-lo[0]), max(0,hi[0]-page_image.shape[1]))
            padded = cv2.copyMakeBorder(page_image, *padding, cv2.BORDER_CONSTANT, value=255)
            lo += [padding[2],padding[0]]
            hi += [padding[2],padding[0]]
            image = cv2.resize(padded[lo[1]:hi[1],lo[0]:hi[0]], (24,24), interpolation=cv2.INTER_AREA)
            if row['augmentation'] == 'horizontal_flip':
                image = cv2.flip(image,1)
            elif row['augmentation'] is not None:
                raise ValueError('unsupported positive augmentation')
            augmentation_provenance.append({'sample_id':row['sample_id'], 'parent_image_path':page_path,
                                            'source_bbox':box.tolist(), 'margin':args.positive_margin,
                                            'augmentation':row['augmentation']})
        else:
            image = cv2.imread(str(args.dataset / row["file"]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (24, 24):
            raise ValueError(f"missing or invalid crop: {row['sample_id']}")
        x.append(hog(image))
        y.append(row["label"])
    proposal_count = 0
    manifest_path = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = ROOT / "models/step4_cascade.json"
    if args.proposal_positive_pages:
        cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
        for page in choose_pages(manifest_path, args.proposal_positive_pages):
            gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(page["image_path"])
            boxes, _, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
            gt = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
            positive = boxes[iou_max(boxes, gt) >= 0.5]
            for vector in candidate_features(gray, positive):
                x.append(vector)
                y.append(1)
            proposal_count += len(positive)
            print(page["image_id"], "positive_proposals", len(positive), flush=True)
    reviewed_negative_count = 0
    if args.reviewed_negative_csv:
        import csv
        with args.reviewed_negative_csv.open(newline="", encoding="utf-8-sig") as stream:
            review = list(csv.DictReader(stream))
        if any(row["review_decision"] not in ("accept_negative", "reject") for row in review):
            raise ValueError("reviewed negatives contain unreviewed candidates")
        for row in review:
            if row["review_decision"] != "accept_negative":
                continue
            if row["split"] != "train":
                raise ValueError("reviewed negative is not from train")
            gray = cv2.imread(str(ROOT / "datasets" / row["parent_image_path"]), cv2.IMREAD_GRAYSCALE)
            box = np.asarray(json.loads(row["bbox_xyxy"]), dtype=np.int32)
            if gray is None or box.shape != (4,) or np.any(box[:2] < 0) or box[2] > gray.shape[1] or box[3] > gray.shape[0]:
                raise ValueError(f"invalid reviewed negative: {row['id']}")
            x.append(candidate_features(gray, box.reshape(1, 4))[0])
            y.append(0)
            reviewed_negative_count += 1
    features = np.asarray(x, dtype=np.float32)
    labels = np.asarray(y, dtype=np.uint8)
    if args.features_output:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_output, x=features, y=labels)
    model = SVC(kernel="rbf", C=args.C, gamma="scale", class_weight="balanced", cache_size=512, random_state=20260925)
    model.fit(features, labels)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model, compress=3)
    report = {"split": "train", "seed": 20260925, "samples": len(labels), "positive": int(labels.sum()),
              "negative": int(len(labels) - labels.sum()), "support_vectors": len(model.support_vectors_),
              "positive_proposals": proposal_count, "reviewed_candidate_negatives": reviewed_negative_count,
              "positive_margin": args.positive_margin,
              "features_sha256": hashlib.sha256(args.features_output.read_bytes()).hexdigest() if args.features_output else None,
              "positive_augmentation_provenance": augmentation_provenance,
              "C": args.C, "gamma": "scale", "kernel": "rbf", "class_weight": "balanced",
              "feature": "324-dimensional 24x24 grayscale HOG, same extractor as current verifier",
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "page_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if args.proposal_positive_pages else None,
              "cascade_sha256": hashlib.sha256(cascade_path.read_bytes()).hexdigest() if args.proposal_positive_pages else None,
              "reviewed_negative_sha256": hashlib.sha256(args.reviewed_negative_csv.read_bytes()).hexdigest() if args.reviewed_negative_csv else None,
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()}
    args.model.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in report.items() if k != 'positive_augmentation_provenance'})


if __name__ == "__main__":
    main()
