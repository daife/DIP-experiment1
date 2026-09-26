#!/usr/bin/env python3
"""Prepare a small, traceable train-page false-positive review batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.candidate_verifier import CandidateVerifier
from src.multiscale import PyramidConfig, detect_multiscale
from train_candidate_verifier import choose_pages


def intersects_annotated_face(box: np.ndarray, annotations: list[dict], pad: float = 0.25) -> bool:
    for annotation in annotations:
        face = np.asarray(annotation["bbox"], dtype=np.float64)
        margin = pad * max(face[2] - face[0], face[3] - face[1])
        if min(box[2], face[2] + margin) > max(box[0], face[0] - margin) and min(box[3], face[3] + margin) > max(box[1], face[1] - margin):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--skip-pages", type=int, default=0)
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--rank-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/derived/step8_false_positive_review_v1")
    args = parser.parse_args()
    if args.pages < 1 or args.per_page < 1 or args.rank_stride < 1 or args.skip_pages < 0:
        parser.error("invalid page, rank, or candidate count")
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
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for page in choose_pages(manifest, args.pages + args.skip_pages)[args.skip_pages:]:
        image_path = ROOT / "datasets" / page["image_path"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        boxes, scores, _ = detect_multiscale(gray, cascade, pyramid)
        hog = verifier.scores(gray, boxes)
        eligible = np.flatnonzero((hog >= threshold) & (boxes[:, 2] - boxes[:, 0] >= min_side))
        eligible = eligible[np.argsort(-scores[eligible], kind="stable")]
        selected = 0
        for rank, index in enumerate(eligible):
            if rank % args.rank_stride:
                continue
            box = boxes[index]
            if intersects_annotated_face(box, page["annotations"]):
                continue
            if any(row["page_id"] == page["image_id"] and
                   np.linalg.norm(np.asarray(row["bbox_xyxy"][:2]) - box[:2]) < 0.5 * (box[2] - box[0]) for row in records):
                continue
            x1, y1, x2, y2 = map(int, box)
            margin = max(16, int(0.5 * max(x2 - x1, y2 - y1)))
            context_x1, context_y1 = max(0, x1-margin), max(0, y1-margin)
            context = cv2.cvtColor(gray[context_y1:min(gray.shape[0], y2+margin),
                                            context_x1:min(gray.shape[1], x2+margin)], cv2.COLOR_GRAY2BGR)
            cv2.rectangle(context, (x1-context_x1, y1-context_y1),
                          (x2-context_x1-1, y2-context_y1-1), (0, 0, 255), 2)
            candidate_id = f"fp{len(records):03d}"
            record = {"id": candidate_id, "page_id": page["image_id"], "parent_image_path": page["image_path"],
                      "bbox_xyxy": [x1, y1, x2, y2], "cascade_score": float(scores[index]),
                      "hog_score": float(hog[index]), "split": "train", "review_decision": "unreviewed",
                      "texture_category": "", "review_note": ""}
            records.append(record)
            cv2.imwrite(str(args.output / f"{candidate_id}_crop.png"), cv2.resize(gray[y1:y2, x1:x2], (96, 96)))
            cv2.imwrite(str(args.output / f"{candidate_id}_context.png"), context)
            panels.append(Image.fromarray(cv2.cvtColor(context, cv2.COLOR_BGR2RGB)))
            selected += 1
            if selected == args.per_page:
                break
        print(page["image_id"], "eligible", len(eligible), "selected", selected, flush=True)
    with (args.output / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["id"])
        writer.writeheader()
        writer.writerows(records)
    if panels:
        cell_w, cell_h, columns = 260, 230, 4
        for start in range(0, len(panels), 24):
            subset = panels[start:start+24]
            sheet = Image.new("RGB", (columns * cell_w, ((len(subset) + columns - 1) // columns) * cell_h), "white")
            draw = ImageDraw.Draw(sheet)
            crop_sheet = Image.new("RGB", (columns * 120, ((len(subset) + columns - 1) // columns) * 120), "white")
            crop_draw = ImageDraw.Draw(crop_sheet)
            for offset, panel in enumerate(subset):
                i = start + offset
                x, y = offset % columns * cell_w, offset // columns * cell_h
                panel.thumbnail((240, 200))
                sheet.paste(panel, (x + (cell_w - panel.width) // 2, y))
                draw.text((x + 8, y + 204), records[i]["id"] + " " + records[i]["page_id"].split(":")[1][:17], fill="black")
                crop = Image.open(args.output / f"{records[i]['id']}_crop.png").convert("RGB")
                cx, cy = offset % columns * 120, offset // columns * 120
                crop_sheet.paste(crop, (cx + 12, cy))
                crop_draw.text((cx + 8, cy + 98), records[i]["id"], fill="black")
            sheet.save(args.output / f"contact_sheet_{start//24:02d}.png")
            crop_sheet.save(args.output / f"crop_sheet_{start//24:02d}.png")
    metadata = {"selection": "top final Cascade scores per fixed train page after current HOG and side gate; exclude padded annotated faces; spatial deduplication",
                "seed": "none; SHA-256 page ordering", "pages": args.pages, "skip_pages": args.skip_pages,
                "per_page": args.per_page, "rank_stride": args.rank_stride,
                "count": len(records), "split": "train", "review_status": "unreviewed",
                "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (manifest, config_path, cascade_path, verifier_path)}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output, len(records))


if __name__ == "__main__":
    main()
