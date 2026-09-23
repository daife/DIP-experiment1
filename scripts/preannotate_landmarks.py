"""Preannotate 28 anime-face landmarks and benchmark throughput.

The anime256 source consists of already-cropped, single-face images.  The
script therefore sends the full crop directly to the HRNetV2 landmark model
instead of running a second face detector over every crop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from anime_face_detector import LandmarkDetector, get_checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "images.csv"
AUTO_ROOT = PROJECT_ROOT / "datasets" / "annotations" / "auto"
MODEL_NAME = "anime-face-detector/hrnetv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=64, help="Number of crops to process.")
    parser.add_argument("--seed", type=int, default=20260923, help="Deterministic sampling seed.")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--warmup", type=int, default=3, help="Processed images excluded from steady-state timing.")
    parser.add_argument("--no-flip-test", action="store_true", help="Disable horizontal-flip test-time averaging.")
    parser.add_argument("--run-name", help="Output directory name; defaults to a UTC timestamp.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def load_pool(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["source_dataset"] == "anime256" and (split == "all" or row["split"] == split)
        ]
    return rows


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return requested


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    pool = load_pool(args.manifest, args.split)
    if not pool:
        raise RuntimeError("No matching anime256 records were found")
    rng = random.Random(args.seed)
    selected = rng.sample(pool, min(args.limit, len(pool)))

    device = resolve_device(args.device)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("hrnetv2_%Y%m%dT%H%M%SZ")
    output_dir = AUTO_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    annotations_path = output_dir / "annotations.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    checkpoint_path = get_checkpoint_path("hrnetv2")
    load_started = time.perf_counter()
    detector = LandmarkDetector(
        checkpoint_path,
        face_detector_name=None,
        device=device,
        flip_test=not args.no_flip_test,
    )
    model_load_seconds = time.perf_counter() - load_started

    durations: list[float] = []
    failures: list[dict[str, str]] = []
    landmark_confidences: list[float] = []
    out_of_bounds_points = 0
    total_started = time.perf_counter()
    face_count = 0
    with annotations_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected:
            item_started = time.perf_counter()
            image_path = PROJECT_ROOT / "datasets" / Path(row["image_path"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                failures.append({"image_id": row["image_id"], "error": "image could not be read"})
                continue

            height, width = image.shape[:2]
            full_crop_box = np.array([0, 0, width - 1, height - 1, 1.0], dtype=np.float32)
            try:
                prediction = detector(image, boxes=[full_crop_box])[0]
            except Exception as exc:  # keep a long batch auditable instead of losing prior output
                failures.append({"image_id": row["image_id"], "error": repr(exc)})
                continue

            keypoints = prediction["keypoints"]
            if keypoints.shape != (28, 3):
                failures.append(
                    {"image_id": row["image_id"], "error": f"unexpected keypoint shape {keypoints.shape}"}
                )
                continue

            landmark_confidences.extend(float(point[2]) for point in keypoints)
            out_of_bounds_points += sum(
                not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height)
                for point in keypoints
            )

            record = {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "source_dataset": row["source_dataset"],
                "source_group": row["source_group"],
                "split": row["split"],
                "width": width,
                "height": height,
                "annotation_source": MODEL_NAME,
                "annotations": [
                    {
                        "bbox": [0.0, 0.0, float(width - 1), float(height - 1)],
                        "bbox_source": "anime256_full_crop",
                        "landmarks": [
                            {
                                "x": float(point[0]),
                                "y": float(point[1]),
                                "visibility": None,
                                "confidence": float(point[2]),
                            }
                            for point in keypoints
                        ],
                    }
                ],
            }
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            face_count += 1
            durations.append(time.perf_counter() - item_started)

    elapsed_seconds = time.perf_counter() - total_started
    excluded = min(args.warmup, max(0, len(durations) - 1))
    steady = durations[excluded:]
    mean_seconds = statistics.fmean(steady)
    total_pool = len(pool)
    all_anime256 = len(load_pool(args.manifest, "all"))

    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": MODEL_NAME,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "flip_test": not args.no_flip_test,
            "input_policy": "anime256 full crop; face detector skipped",
        },
        "sampling": {
            "manifest": str(args.manifest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_dataset": "anime256",
            "split": args.split,
            "pool_size": total_pool,
            "seed": args.seed,
            "requested": args.limit,
            "selected": len(selected),
            "warmup_excluded": excluded,
        },
        "environment": {
            "device": device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "result": {
            "successful_images": len(durations),
            "annotated_faces": face_count,
            "failures": failures,
            "model_load_seconds": model_load_seconds,
            "batch_elapsed_seconds": elapsed_seconds,
            "steady_mean_seconds_per_image": mean_seconds,
            "steady_median_seconds_per_image": statistics.median(steady),
            "steady_p95_seconds_per_image": percentile(steady, 0.95),
            "steady_images_per_second": 1.0 / mean_seconds,
            "estimated_current_pool_seconds": total_pool * mean_seconds,
            "estimated_all_anime256_seconds": all_anime256 * mean_seconds,
            "landmark_points": len(landmark_confidences),
            "out_of_bounds_landmark_points": out_of_bounds_points,
            "landmark_confidence_mean": statistics.fmean(landmark_confidences),
            "landmark_confidence_median": statistics.median(landmark_confidences),
            "landmark_confidence_min": min(landmark_confidences),
            "landmark_confidence_max": max(landmark_confidences),
        },
        "outputs": {"annotations": annotations_path.name},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
