#!/usr/bin/env python3
"""Compare pyramid factors and scan steps on fixed validation-page crops."""

from __future__ import annotations

import argparse
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
from src.multiscale import PyramidConfig, detect_multiscale  # noqa: E402


def choose_pages(manifest: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    eligible = [row for row in rows if row["split"] == "validation" and row["annotations"]]
    eligible.sort(key=lambda row: hashlib.sha256(("step5-comparison:" + row["image_id"]).encode()).digest())
    selected, groups = [], set()
    for row in eligible:
        if row["source_group"] not in groups:
            selected.append(row)
            groups.add(row["source_group"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} distinct validation groups available")
    return selected


def face_crop(gray: np.ndarray, annotations: list[dict], side: int) -> tuple[np.ndarray, list[int], list[int]]:
    height, width = gray.shape
    if min(width, height) < side:
        raise ValueError("page is smaller than requested crop")
    fitting = [item for item in annotations if
               24 <= max(item["bbox"][2] - item["bbox"][0], item["bbox"][3] - item["bbox"][1]) <= side * 0.6]
    if not fitting:
        raise ValueError("page has no face of suitable size for the comparison crop")
    face = max(fitting, key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]))["bbox"]
    center_x, center_y = (face[0] + face[2]) // 2, (face[1] + face[3]) // 2
    x1 = min(max(0, center_x - side // 2), width - side)
    y1 = min(max(0, center_y - side // 2), height - side)
    return gray[y1:y1 + side, x1:x1 + side], [x1, y1, x1 + side, y1 + side], face


def best_iou(boxes: np.ndarray, face: list[int]) -> float:
    if not len(boxes):
        return 0.0
    overlap = np.maximum(0, np.minimum(boxes[:, 2:], face[2:]) - np.maximum(boxes[:, :2], face[:2]))
    intersection = overlap[:, 0].astype(np.float64) * overlap[:, 1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    face_area = (face[2] - face[0]) * (face[3] - face[1])
    return float(np.max(intersection / (areas + face_area - intersection)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/step4_cascade.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/normalized/manga109_faces.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/step5_comparison.json")
    parser.add_argument("--full-page-output", type=Path, default=ROOT / "results/step5_full_page.json")
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "datasets/derived/step5_comparison")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    args = parser.parse_args()
    if args.pages < 1 or args.crop_size < 24:
        raise ValueError("pages must be positive and crop-size at least 24")
    model_bytes = args.model.read_bytes()
    model = cascade_from_dict(json.loads(model_bytes))
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    report = {"model_sha256": hashlib.sha256(model_bytes).hexdigest(),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "selection": "SHA-256 step5-comparison:image_id order; distinct validation source groups; largest annotated face with max side 24..0.6*crop-size, centered",
              "crop_size": args.crop_size, "nms_iou": args.nms_iou, "runs": []}
    selected_pages = choose_pages(args.manifest, args.pages)
    for page in selected_pages:
        image_path = ROOT / "datasets" / page["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        crop, crop_box, anchor_face = face_crop(image, page["annotations"], args.crop_size)
        face_in_crop = [anchor_face[0] - crop_box[0], anchor_face[1] - crop_box[1],
                        anchor_face[2] - crop_box[0], anchor_face[3] - crop_box[1]]
        for scale_factor in (1.1, 1.2, 1.3):
            for step in (1, 2):
                start = perf_counter()
                boxes, scores, layers = detect_multiscale(crop, model, PyramidConfig(scale_factor, step, args.nms_iou))
                seconds = perf_counter() - start
                row = {"image_id": page["image_id"], "source_group": page["source_group"],
                       "crop_xyxy": crop_box, "anchor_face_xyxy": anchor_face,
                       "scale_factor": scale_factor, "step": step, "layers": layers,
                       "total_windows": sum(layer["windows"] for layer in layers),
                       "total_candidates": sum(layer["candidates"] for layer in layers),
                       "nms_detections": len(boxes), "total_seconds": seconds,
                       "anchor_best_iou": best_iou(boxes, face_in_crop),
                       "boxes_xyxy_in_crop": boxes.tolist(), "scores": scores.tolist()}
                report["runs"].append(row)
                preview = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                for x1, y1, x2, y2 in boxes[:20]:
                    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2 - 1), int(y2 - 1)), (0, 0, 255), 1)
                cv2.rectangle(preview, tuple(face_in_crop[:2]), tuple(face_in_crop[2:]), (0, 255, 0), 2)
                preview_name = f'{page["image_id"].replace(":", "_")}_scale{scale_factor}_step{step}.png'
                cv2.imwrite(str(args.preview_dir / preview_name), preview)
                print(f'{page["image_id"]} scale={scale_factor} step={step}: '
                      f'{row["total_windows"]} windows, {row["total_candidates"]} candidates, '
                      f'{len(boxes)} after NMS, {seconds:.2f}s', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    page = selected_pages[0]
    image_path = ROOT / "datasets" / page["image_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    full_page = {"image_id": page["image_id"], "image_path": page["image_path"],
                 "source_group": page["source_group"], "split": page["split"],
                 "image_size": [image.shape[1], image.shape[0]],
                 "model_sha256": report["model_sha256"], "manifest_sha256": report["manifest_sha256"],
                 "scale_factor": 1.2, "nms_iou": args.nms_iou, "runs": []}
    for step in (1, 2):
        start = perf_counter()
        boxes, scores, layers = detect_multiscale(image, model, PyramidConfig(1.2, step, args.nms_iou))
        seconds = perf_counter() - start
        full_page["runs"].append({"step": step, "layers": layers,
                                  "total_windows": sum(layer["windows"] for layer in layers),
                                  "total_candidates": sum(layer["candidates"] for layer in layers),
                                  "nms_detections": len(boxes), "total_seconds": seconds,
                                  "boxes_xyxy_in_original_page": boxes.tolist(), "scores": scores.tolist()})
        print(f'full page {page["image_id"]} step={step}: '
              f'{full_page["runs"][-1]["total_windows"]} windows, '
              f'{full_page["runs"][-1]["total_candidates"]} candidates, '
              f'{len(boxes)} after NMS, {seconds:.2f}s', flush=True)
    args.full_page_output.parent.mkdir(parents=True, exist_ok=True)
    args.full_page_output.write_text(json.dumps(full_page, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
