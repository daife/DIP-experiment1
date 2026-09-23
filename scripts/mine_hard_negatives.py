#!/usr/bin/env python3
"""Scan train Manga109 pages and collect Cascade false positives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade import cascade_from_dict  # noqa: E402
from src.channels11 import compute_11_channels  # noqa: E402
from src.page_scan import scan_page  # noqa: E402


def overlaps_face(box: list[int], faces: list[list[int]]) -> bool:
    x1, y1, x2, y2 = box
    for fx1, fy1, fx2, fy2 in faces:
        pad = 0.125 * max(fx2 - fx1, fy2 - fy1)
        if min(x2, fx2 + pad) > max(x1, fx1 - pad) and min(y2, fy2 + pad) > max(y1, fy1 - pad):
            return True
    return False


def overlap_ratio(a: list[int], b: list[int]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def choose_pages(manifest: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    eligible = [row for row in rows if row["split"] == "train" and row["annotations"]]
    eligible.sort(key=lambda row: hashlib.sha256(("step4-mining:" + row["image_id"]).encode()).digest())
    selected = []
    groups = set()
    for row in eligible:
        if row["source_group"] in groups:
            continue
        selected.append(row)
        groups.add(row["source_group"])
        if len(selected) == limit:
            break
    return selected


def write_contact_sheet(records: list[dict], crops: list[Image.Image], destination: Path) -> None:
    cell = 116
    columns = 6
    rows = (min(len(crops), 24) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for i, crop in enumerate(crops[:24]):
        x, y = (i % columns) * cell, (i // columns) * cell
        sheet.paste(crop.resize((88, 88), Image.Resampling.NEAREST).convert("RGB"), (x + 14, y))
        draw.text((x + 5, y + 91), records[i]["id"], fill="black", font=ImageFont.load_default())
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/step4_initial_cascade.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/normalized/manga109_faces.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/derived/step4_hard_negatives_v1")
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--max-crops", type=int, default=240)
    parser.add_argument("--max-page-dimension", type=int, default=768)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()
    if min(args.pages, args.max_crops, args.max_page_dimension, args.step) < 1:
        raise ValueError("all count and size options must be positive")
    model_bytes = args.model.read_bytes()
    model = cascade_from_dict(json.loads(model_bytes.decode("utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[float, dict, list[int]]] = []
    for page in choose_pages(args.manifest, args.pages):
        with Image.open(ROOT / "datasets" / page["image_path"]) as source:
            original = source.convert("L")
        ratio = min(1.0, args.max_page_dimension / max(original.size))
        new_size = (max(24, round(original.width * ratio)), max(24, round(original.height * ratio)))
        scaled = original.resize(new_size, Image.Resampling.LANCZOS)
        boxes, scores = scan_page(np.asarray(scaled), model, step=args.step)
        faces = [annotation["bbox"] for annotation in page["annotations"]]
        # Rank on the final-stage score; cap per page to preserve diversity.
        accepted = 0
        page_boxes: list[list[int]] = []
        for index in np.argsort(scores)[::-1]:
            box = boxes[index]
            original_box = [round(box[0] * original.width / scaled.width),
                            round(box[1] * original.height / scaled.height),
                            round(box[2] * original.width / scaled.width),
                            round(box[3] * original.height / scaled.height)]
            if overlaps_face(original_box, faces) or any(overlap_ratio(original_box, existing) > 0.3 for existing in page_boxes):
                continue
            candidates.append((float(scores[index]), page, original_box))
            page_boxes.append(original_box)
            accepted += 1
            if accepted >= 20:
                break
        print(f'{page["image_id"]}: {len(boxes)} passed, {accepted} eligible', flush=True)
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    candidates = candidates[:args.max_crops]
    records, crops, channels = [], [], []
    for index, (score, page, box) in enumerate(candidates):
        with Image.open(ROOT / "datasets" / page["image_path"]) as source:
            crop = source.convert("L").crop(tuple(box)).resize((24, 24), Image.Resampling.LANCZOS)
        item = {"id": f"hn{index:04d}", "page_id": page["image_id"],
                "parent_image_path": page["image_path"], "bbox_xyxy": box,
                "score": score, "split": "train", "label": 0,
                "review_decision": "unreviewed"}
        records.append(item)
        crops.append(crop)
        channels.append(np.stack(compute_11_channels(np.asarray(crop))))
        crop.save(args.output / f'{item["id"]}.png')
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    np.save(args.output / "channels11.npy", np.stack(channels) if channels else np.empty((0, 11, 24, 24), dtype=np.uint8))
    if crops:
        write_contact_sheet(records, crops, args.output / "contact_sheet.png")
    metadata = {
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "pages_scanned": len(choose_pages(args.manifest, args.pages)),
        "max_crops": args.max_crops,
        "max_page_dimension": args.max_page_dimension,
        "step": args.step,
        "selected_candidates": len(records),
        "selection": "highest final-stage scores, no annotated-face overlap, per-page IoU <= 0.3",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages_scanned": len(choose_pages(args.manifest, args.pages)),
                      "candidates": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
