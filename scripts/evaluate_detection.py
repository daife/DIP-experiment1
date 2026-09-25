#!/usr/bin/env python3
"""Evaluate fixed validation pages with score-ordered one-to-one IoU matching."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict  # noqa: E402
from src.detection_metrics import match_detections  # noqa: E402
from src.multiscale import PyramidConfig, detect_multiscale  # noqa: E402
from compare_multiscale import choose_pages  # noqa: E402


def main() -> None:
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    model_path = ROOT / "models/step4_cascade.json"
    previous_path = ROOT / "results/step5_full_page.json"
    model_bytes = model_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    model = cascade_from_dict(json.loads(model_bytes))
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous["model_sha256"] != model_hash or previous["manifest_sha256"] != manifest_hash:
        raise ValueError("step5 cached page uses a different model or manifest")
    output = {"model_sha256": model_hash, "manifest_sha256": manifest_hash,
              "split": "validation", "selection": "step5-comparison SHA-256 order, two distinct source groups",
              "scale_factor": 1.2, "nms_iou": 0.3, "match_iou": 0.5,
              "box_convention": "xyxy half-open", "pages": [], "summary": {}}
    for page in choose_pages(manifest, 2):
        image_path = ROOT / "datasets" / page["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        ground_truth = np.asarray([item["bbox"] for item in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        page_out = {"image_id": page["image_id"], "image_path": page["image_path"],
                    "source_group": page["source_group"], "size": [image.shape[1], image.shape[0]],
                    "gt_boxes": ground_truth.tolist(), "runs": []}
        for step in (1, 2):
            if page["image_id"] == previous["image_id"]:
                cached = next(run for run in previous["runs"] if run["step"] == step)
                boxes = np.asarray(cached["boxes_xyxy_in_original_page"], dtype=np.int32).reshape(-1, 4)
                scores = np.asarray(cached["scores"], dtype=np.float64)
                layers = cached["layers"]
                seconds = cached["total_seconds"]
                timing_source = "step5_full_page.json (2026-09-23 run)"
            else:
                start = perf_counter()
                boxes, scores, layers = detect_multiscale(image, model, PyramidConfig(1.2, step, 0.3))
                seconds = perf_counter() - start
                timing_source = "current run"
            metrics = match_detections(boxes, scores, ground_truth, 0.5)
            page_out["runs"].append({"step": step, "timing_source": timing_source,
                "seconds": seconds, "windows": sum(layer["windows"] for layer in layers),
                "cascade_candidates": sum(layer["candidates"] for layer in layers),
                "nms_detections": len(boxes), "metrics": metrics,
                "boxes": boxes.tolist(), "scores": scores.tolist(), "layers": layers})
            print(f'{page["image_id"]} step={step}: TP={metrics["tp"]} FP={metrics["fp"]} '
                  f'FN={metrics["fn"]} {seconds:.2f}s', flush=True)
        output["pages"].append(page_out)
    for step in (1, 2):
        runs = [p["runs"][step - 1] for p in output["pages"]]
        tp = sum(run["metrics"]["tp"] for run in runs)
        fp = sum(run["metrics"]["fp"] for run in runs)
        fn = sum(run["metrics"]["fn"] for run in runs)
        output["summary"][f"step{step}"] = {
            "pages": len(runs), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "mean_seconds_per_page": sum(run["seconds"] for run in runs) / len(runs),
            "total_windows": sum(run["windows"] for run in runs),
            "mean_windows_per_page": sum(run["windows"] for run in runs) / len(runs)}
    target = ROOT / "results/step7_detection_evaluation.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
