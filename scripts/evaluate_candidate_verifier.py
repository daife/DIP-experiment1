#!/usr/bin/env python3
"""Compare baseline and fixed HOG gates on distinct validation works."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.detection_metrics import match_detections
from src.multiscale import PyramidConfig, detect_multiscale
from src.candidate_verifier import features
from compare_multiscale import choose_pages


def choose_test_pages(manifest: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == "test" and row["annotations"]]
    rows.sort(key=lambda row: hashlib.sha256(("step8-test:" + row["image_id"]).encode()).digest())
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
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--verifier", type=Path, default=ROOT / "models/candidate_hog_svm.npz")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = ROOT / "models/step4_cascade.json"
    verifier_path = args.verifier
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    with np.load(verifier_path) as data:
        coefficients, intercept = data["coefficients"], float(data["intercept"])
    rules = ({"baseline": (None, None), "hog0.5_side48": (0.5, 48),
              "hog0.75_side48": (0.75, 48), "hog0.75_side36": (0.75, 36),
              "hog0_side80": (0.0, 80)} if args.split == "validation" else
             {"baseline": (None, None), "hog0.75_side36": (0.75, 36)})
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in rules}
    pages = []
    selected = choose_pages(manifest, 8) if args.split == "validation" else choose_test_pages(manifest, 8)
    for page in selected:
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        gt = np.asarray([a["bbox"] for a in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        boxes, scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
        hog_scores = features(gray, boxes) @ coefficients + intercept
        item = {"id": page["image_id"], "gt": len(gt), "candidates": len(boxes), "rules": {}}
        for name, (threshold, side) in rules.items():
            keep = np.ones(len(boxes), dtype=bool) if threshold is None else ((hog_scores >= threshold) & (boxes[:, 2] - boxes[:, 0] >= side))
            result = match_detections(boxes[keep], scores[keep], gt, 0.5)
            item["rules"][name] = {k: result[k] for k in ("tp", "fp", "fn")}
            for key in ("tp", "fp", "fn"):
                totals[name][key] += result[key]
        pages.append(item)
        print(item, flush=True)
    for result in totals.values():
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        result["precision"] = tp / (tp + fp) if tp + fp else 0
        result["recall"] = tp / (tp + fn) if tp + fn else 0
        result["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    output = {"split": args.split, "step": 2, "scale_factor": 1.2, "nms_iou": 0.3,
              "cascade_sha256": hashlib.sha256(cascade_path.read_bytes()).hexdigest(),
              "verifier_sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "selection": ("step5-comparison" if args.split == "validation" else "step8-test") + " SHA-256 order, eight distinct source groups",
              "pages": pages, "summary": totals}
    target = args.output or ROOT / f"results/step8_candidate_verifier_{args.split}.json"
    target.write_bytes((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(totals)


if __name__ == "__main__":
    main()
