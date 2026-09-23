#!/usr/bin/env python3
"""Build reproducible 24x24 detection crops from Manga109 pages.

The source manifest assigns splits by manga title. This script never re-splits
images. Human review decisions can be supplied with --review-csv and are
applied to the exported sample manifest without editing source annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SEED = "experiment1-step3-v1"
TARGETS = {"train": 1800, "validation": 240, "test": 360}
SIZE = 24
REVIEW_FIELDS = ["sample_id", "decision", "tags", "note"]
TAGS = {"face", "background", "hair", "clothing", "hand", "text_box", "sound_effect", "building", "object", "face_like", "blurred", "bad_crop", "occlusion", "rotation", "other"}


def stable_int(*parts: object) -> int:
    payload = ":".join(map(str, (SEED, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_records(path: Path) -> dict[str, list[dict]]:
    by_split = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            by_split[record["split"]].append(record)
    return by_split


def ordered_round_robin(records: list[dict], key: str) -> list[tuple[dict, int]]:
    groups = defaultdict(list)
    for record in records:
        for index, annotation in enumerate(record["annotations"]):
            x1, y1, x2, y2 = annotation["bbox"]
            if min(x2 - x1, y2 - y1) >= 24:
                groups[record["source_group"]].append((record, index))
    for group, items in groups.items():
        items.sort(key=lambda item: stable_int(key, item[0]["image_id"], item[1]))
    titles = sorted(groups, key=lambda group: stable_int(key, group))
    result = []
    while True:
        added = False
        for title in titles:
            if groups[title]:
                result.append(groups[title].pop())
                added = True
        if not added:
            return result


def square_box(box: list[int], margin: float = 1.15) -> list[int]:
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * margin
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = math.floor(cx - side / 2), math.floor(cy - side / 2)
    side = math.ceil(side)
    return [left, top, left + side, top + side]


def crop_square(image: Image.Image, box: list[int]) -> Image.Image:
    # PIL crops outside an image with zero fill. Manga page backgrounds are white.
    x1, y1, x2, y2 = box
    canvas = Image.new("L", (x2 - x1, y2 - y1), 255)
    common = [max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)]
    if common[0] < common[2] and common[1] < common[3]:
        canvas.paste(image.crop(tuple(common)), (common[0] - x1, common[1] - y1))
    return canvas.resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def near_face(box: list[int], faces: list[list[int]]) -> bool:
    # Zero overlap with boxes expanded by 25% prevents even a full tiny face
    # being included in a large negative window.
    x1, y1, x2, y2 = box
    for face in faces:
        fx1, fy1, fx2, fy2 = face
        pad = 0.125 * max(fx2 - fx1, fy2 - fy1)
        if min(x2, fx2 + pad) > max(x1, fx1 - pad) and min(y2, fy2 + pad) > max(y1, fy1 - pad):
            return True
    return False


def sample_negatives(records: list[dict], target: int, split: str) -> list[tuple[dict, list[int]]]:
    # Prefer a wide selection of pages and titles. Each page contributes at
    # most two windows. Candidate sizes span local detail to larger scenery.
    pages = sorted(records, key=lambda row: stable_int("page", row["image_id"]))
    result = []
    used = set()
    for pass_index in range(2):
        for record in pages:
            if len(result) >= target:
                return result
            width, height = record["width"], record["height"]
            rng = random.Random(stable_int("negative", record["image_id"], pass_index))
            max_side = min(width, height, 320)
            if max_side < 24:
                continue
            faces = [ann["bbox"] for ann in record["annotations"]]
            source = ROOT / "datasets" / record["image_path"]
            with Image.open(source) as image:
                gray = image.convert("L")
                for _ in range(100):
                    side = min(max_side, rng.choice([32, 48, 64, 96, 128, 192, 256, 320]))
                    x = rng.randrange(0, width - side + 1)
                    y = rng.randrange(0, height - side + 1)
                    box = [x, y, x + side, y + side]
                    if near_face(box, faces):
                        continue
                    key = (record["image_id"], *box)
                    if key in used:
                        continue
                    thumbnail = np.asarray(gray.crop(tuple(box)).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
                    ink_fraction = float(np.mean(thumbnail < 220))
                    # Reject blank margins and nearly solid black panels.
                    if thumbnail.std() < 24 or not 0.04 <= ink_fraction <= 0.85:
                        continue
                    used.add(key)
                    result.append((record, box))
                    break
    if len(result) < target:
        raise RuntimeError(f"Only {len(result)} safe negative windows found for {split}; wanted {target}")
    return result


def write_sheet(rows: list[dict], image_root: Path, destination: Path, columns: int = 10) -> None:
    cell_w, cell_h = 120, 100
    sheet = Image.new("RGB", (columns * cell_w, math.ceil(len(rows) / columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x, y = (index % columns) * cell_w, (index // columns) * cell_h
        with Image.open(image_root / row["file"]) as source:
            preview = source.convert("RGB").resize((72, 72), Image.Resampling.NEAREST)
        sheet.paste(preview, (x + 24, y + 2))
        draw.text((x + 4, y + 77), row["sample_id"], fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def load_review(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row["sample_id"].strip()
            decision = row["decision"].strip().lower()
            tags = [part.strip().lower() for part in row["tags"].split(";") if part.strip()]
            if decision not in {"", "keep", "reject"}:
                raise ValueError(f"Bad decision for {sample_id}: {decision}")
            if any(tag not in TAGS for tag in tags):
                raise ValueError(f"Bad tag for {sample_id}: {tags}")
            if sample_id in result:
                raise ValueError(f"Duplicate review ID: {sample_id}")
            result[sample_id] = {"decision": decision, "tags": tags, "note": row["note"].strip()}
    return result


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest)
    review = load_review(args.review_csv)
    rows = []
    for split in ("train", "validation", "test"):
        target = args.train if split == "train" else args.validation if split == "validation" else args.test
        positives = ordered_round_robin(records[split], "positive")[:target]
        if len(positives) < target:
            raise RuntimeError(f"Only {len(positives)} positive boxes for {split}; wanted {target}")
        negatives = sample_negatives(records[split], target, split)
        for label, selected in ((1, positives), (0, negatives)):
            for index, item in enumerate(selected):
                record, annotation_index = item if label else (item[0], None)
                source_box = record["annotations"][annotation_index]["bbox"] if label else None
                box = square_box(source_box) if label else item[1]
                prefix = {"train": "tr", "validation": "va", "test": "te"}[split]
                sample_id = f"{prefix}{'p' if label else 'n'}{index:05d}"
                relative = f"{split}/{'positive' if label else 'negative'}/{sample_id}.png"
                source = ROOT / "datasets" / record["image_path"]
                with Image.open(source) as image:
                    crop = crop_square(image.convert("L"), box)
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path)
                rows.append({
                    "sample_id": sample_id, "file": relative, "label": label, "split": split,
                    "parent_image_id": record["image_id"], "parent_image_path": record["image_path"],
                    "source_dataset": record["source_dataset"], "source_group": record["source_group"],
                    "source_bbox_xyxy": source_box, "crop_bbox_xyxy": box,
                    "annotation_index": annotation_index,
                    "annotation_source": record["annotation_source"] if label else None,
                    "augmentation": None,
                })
                # Train-only deterministic augmentation. Apply equally to both
                # classes to retain class balance. No augmented crop gets a new split.
                if split == "train" and args.augment:
                    augmented_id = sample_id + "f"
                    augmented_relative = f"{split}/{'positive' if label else 'negative'}/{augmented_id}.png"
                    ImageOps.mirror(crop).save(output / augmented_relative)
                    rows.append({**rows[-1], "sample_id": augmented_id, "file": augmented_relative,
                                 "augmentation": "horizontal_flip", "parent_sample_id": sample_id})
    base = [row for row in rows if row["augmentation"] is None]
    known = {row["sample_id"] for row in base}
    unknown = set(review) - known
    if unknown:
        raise ValueError(f"Review contains unknown IDs: {sorted(unknown)[:10]}")
    for row in rows:
        key = row.get("parent_sample_id", row["sample_id"])
        decision = review.get(key, {"decision": "", "tags": [], "note": ""})
        row["review_decision"] = decision["decision"] or "unreviewed"
        row["review_tags"] = decision["tags"]
        row["review_note"] = decision["note"]
    with (output / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output / "split_assignment.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parent_image_id", "source_group", "split"])
        for record in sorted((row for split_rows in records.values() for row in split_rows), key=lambda r: r["image_id"]):
            writer.writerow([record["image_id"], record["source_group"], record["split"]])
    sheets = output / "review_sheets"
    for split in ("train", "validation", "test"):
        for label, name in ((1, "positive"), (0, "negative")):
            items = [row for row in base if row["split"] == split and row["label"] == label]
            for start in range(0, len(items), 100):
                write_sheet(items[start:start + 100], output, sheets / f"{split}_{name}_{start // 100 + 1:03d}.jpg")
    with (output / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in base:
            writer.writerow({"sample_id": row["sample_id"], "decision": "", "tags": "", "note": ""})
    stats = {"seed": SEED, "size": SIZE, "target_per_class": {"train": args.train, "validation": args.validation, "test": args.test},
             "base_count": len(base), "augmented_count": len(rows) - len(base),
             "reviewed_count": sum(row["review_decision"] != "unreviewed" for row in base),
             "rejected_count": sum(row["review_decision"] == "reject" for row in base),
             "manifest": str(args.manifest.relative_to(ROOT))}
    (output / "build_info.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/normalized/manga109_faces.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--train", type=int, default=TARGETS["train"])
    parser.add_argument("--validation", type=int, default=TARGETS["validation"])
    parser.add_argument("--test", type=int, default=TARGETS["test"])
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
