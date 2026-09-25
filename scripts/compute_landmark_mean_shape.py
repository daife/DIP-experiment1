"""Compute a train-only mean shape from the accepted coarse 28-point review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets/annotations/corrected/landmark28_review320.jsonl"
SCHEMA = ROOT / "datasets/annotations/landmark28_schema.json"
OUTPUT = ROOT / "results/landmark_mean_shape_coarse.json"


def compute(records: list[dict]) -> dict:
    sums = np.zeros((28, 2), dtype=np.float64)
    counts = np.zeros(28, dtype=np.int64)
    used = 0
    for record in records:
        if record["split"] != "train":
            continue
        if record["review"]["status"] != "human_coarse_review_accepted":
            raise ValueError(f"Unaccepted record: {record['image_id']}")
        annotations = record["annotations"]
        if len(annotations) != 1 or len(annotations[0]["landmarks"]) != 28:
            raise ValueError(f"Expected one face with 28 points: {record['image_id']}")
        x0, y0, x1, y1 = annotations[0]["bbox"]
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox: {record['image_id']}")
        any_visible = False
        for i, point in enumerate(annotations[0]["landmarks"]):
            if point["visibility"] != 1:
                continue
            if point["review_type"] != "V":
                raise ValueError(f"Visible point type mismatch: {record['image_id']}:{i}")
            xy = np.array([(point["x"] - x0) / (x1 - x0), (point["y"] - y0) / (y1 - y0)])
            if not np.isfinite(xy).all():
                raise ValueError(f"Nonfinite point: {record['image_id']}:{i}")
            sums[i] += xy
            counts[i] += 1
            any_visible = True
        used += any_visible
    if np.any(counts == 0):
        raise ValueError(f"No visible training samples for indices {np.where(counts == 0)[0].tolist()}")
    return {
        "schema_id": json.loads(SCHEMA.read_text(encoding="utf-8"))["schema_id"],
        "source_annotations": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "split": "train",
        "review_level": "human_coarse_review_accepted; original HRNetV2 coordinates unchanged",
        "normalization": "((x - bbox_x0) / (bbox_x1 - bbox_x0), (y - bbox_y0) / (bbox_y1 - bbox_y0))",
        "point_policy": "visibility=1 only; H and U excluded; each point averaged over available train images",
        "usable_train_images": int(used),
        "point_counts": counts.tolist(),
        "points_xy": (sums / counts[:, None]).tolist(),
    }


def main() -> None:
    records = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = compute(records)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}; {result['usable_train_images']} train images; point counts {min(result['point_counts'])}–{max(result['point_counts'])}")


if __name__ == "__main__":
    main()
