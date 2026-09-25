"""Train and evaluate fixed-split four-stage landmark regressors."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landmark_regression import LandmarkRegressor, make_offsets, normalized_image, train  # noqa: E402

ANNOTATIONS = ROOT / "datasets/annotations/corrected/landmark28_review320.jsonl"
MEAN = ROOT / "results/landmark_mean_shape_coarse.json"
OUTPUT = ROOT / "results/landmark_regression_coarse_metrics.json"
MODEL_DIR = ROOT / "models"
SEED = 20260925


def load_data() -> dict[str, dict]:
    sets = {s: {"images": [], "targets": [], "visible": [], "reliable": [], "bboxes": [], "ids": []}
            for s in ("train", "validation", "test")}
    for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["review"]["status"] != "human_coarse_review_accepted":
            raise ValueError(f"Unaccepted review: {record['image_id']}")
        annotation = record["annotations"][0]
        bbox = np.asarray(annotation["bbox"], dtype=np.float64)
        points = annotation["landmarks"]
        path = ROOT / "datasets" / record["image_path"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None or len(points) != 28:
            raise ValueError(f"Missing image or invalid points: {record['image_id']}")
        item = sets[record["split"]]
        item["images"].append(normalized_image(gray, bbox))
        target = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float64)
        item["targets"].append((target - bbox[:2]) / (bbox[2:] - bbox[:2]))
        item["visible"].append([p["review_type"] == "V" for p in points])
        item["reliable"].append([p["review_type"] == "H" for p in points])
        item["bboxes"].append(bbox)
        item["ids"].append(record["image_id"])
    for item in sets.values():
        for key in ("images", "targets", "visible", "reliable", "bboxes"):
            item[key] = np.asarray(item[key])
    return sets


def evaluate(model: LandmarkRegressor, data: dict) -> dict:
    pred = model.predict_normalized(data["images"])
    bbox = data["bboxes"]
    wh = bbox[:, 2:] - bbox[:, :2]
    distances = np.linalg.norm((pred - data["targets"]) * wh[:, None, :], axis=2)
    visible = data["visible"]
    reliable = data["reliable"]
    # Schema: image-left eye 11:17, image-right eye 17:23.
    eye_dist = np.empty(len(pred), dtype=np.float64)
    fallback = 0
    for i in range(len(pred)):
        left = np.flatnonzero(visible[i, 11:17]) + 11
        right = np.flatnonzero(visible[i, 17:23]) + 17
        if len(left) and len(right):
            l = (data["targets"][i, left] * wh[i]).mean(axis=0)
            r = (data["targets"][i, right] * wh[i]).mean(axis=0)
            eye_dist[i] = np.linalg.norm(l - r)
        else:
            eye_dist[i] = np.linalg.norm(wh[i])
            fallback += 1
    if np.any(eye_dist <= 0):
        raise ValueError("Nonpositive NME denominator")
    normalized = distances / eye_dist[:, None]
    per_image = []
    for i, image_id in enumerate(data["ids"]):
        per_image.append({"image_id": image_id, "visible_points": int(visible[i].sum()),
                          "visible_nme": float(normalized[i, visible[i]].mean()) if visible[i].any() else None,
                          "reliable_occluded_points": int(reliable[i].sum()),
                          "reliable_occluded_error": float(normalized[i, reliable[i]].mean()) if reliable[i].any() else None,
                          "normalizer": "eye_centers" if visible[i, 11:17].any() and visible[i, 17:23].any() else "bbox_diagonal"})
    return {"images": len(pred), "visible_points": int(visible.sum()),
            "visible_nme": float(normalized[visible].mean()),
            "reliable_occluded_points": int(reliable.sum()),
            "reliable_occluded_error": float(normalized[reliable].mean()) if reliable.any() else None,
            "bbox_diagonal_fallback_images": fallback, "per_image": per_image}


def main() -> None:
    data = load_data()
    mean_data = json.loads(MEAN.read_text(encoding="utf-8"))
    if mean_data["source_sha256"] != hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest():
        raise ValueError("Mean shape source does not match annotations")
    mean = np.asarray(mean_data["points_xy"], dtype=np.float64)
    offsets = make_offsets(SEED)
    baseline = LandmarkRegressor(mean, offsets, np.empty((0, 225, 56)))
    result = {"annotation_sha256": mean_data["source_sha256"], "mean_shape_sha256": hashlib.sha256(MEAN.read_bytes()).hexdigest(),
              "seed": SEED, "stages": 4, "differences_per_point": 8, "feature_count": 224,
              "offset_radius_bbox_fraction": 0.075, "ridge_alpha": 10.0,
              "image_normalization_size": 256,
              "nme_definition": "mean visible-point Euclidean error / visible eye-center distance; bbox diagonal fallback",
              "target_status": "coarse-human-accepted, unchanged HRNetV2 coordinates",
              "baseline": {split: evaluate(baseline, data[split]) for split in ("validation", "test")}, "models": {}}
    MODEL_DIR.mkdir(exist_ok=True)
    for name, include_h in (("visible_only", False), ("visible_and_reliable_occluded", True)):
        mask = data["train"]["visible"] | (data["train"]["reliable"] if include_h else False)
        model = train(data["train"]["images"], data["train"]["targets"], mask, mean, offsets, stages=4, alpha=10.0)
        path = MODEL_DIR / f"landmark_ridge4_{name}.npz"
        model.save(path)
        restored = LandmarkRegressor.load(path)
        result["models"][name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "train_target_points": int(mask.sum()),
                                   "train_target_images": int(mask.any(axis=1).sum()),
                                   "validation": evaluate(restored, data["validation"]),
                                   "test": evaluate(restored, data["test"])}
        print(name, "validation", result["models"][name]["validation"]["visible_nme"],
              "test", result["models"][name]["test"]["visible_nme"], flush=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
