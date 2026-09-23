#!/usr/bin/env python3
"""Build reproducible inventories and normalized face-box manifests.

The source datasets under ``raw`` are read-only inputs. All generated files are
written under ``manifests``. Splits are deterministic and assigned by source
group so that related pages/crops cannot cross train, validation, and test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_SEED = "experiment1-anime-face-v1"
LAP_GROUP_RE = re.compile(r"^(MAL\d{5})_")


def stable_split(dataset: str, group: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{dataset}:{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.75:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def dump_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def jsonl_stats(path: Path) -> dict[str, int]:
    records = boxes = positive_records = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            count = len(record["annotations"])
            records += 1
            boxes += count
            positive_records += count > 0
    return {"records": records, "positive_records": positive_records, "face_boxes": boxes}


def parse_lap_labels(path: Path, kind: str):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            raw = raw.strip()
            if not raw:
                continue
            image_part, payload = raw.split(":", 1)
            relative = image_part.strip().replace("\\", "/")
            annotations = []
            if payload.strip() != "-":
                for item in payload.split(","):
                    values = item.split()
                    if len(values) != 5:
                        raise ValueError(f"{path}:{line_number}: expected five values: {item!r}")
                    x1, y1, x2, y2 = map(int, values[:4])
                    annotation = {"bbox": [x1, y1, x2, y2], "category": "face"}
                    if kind == "manual":
                        annotation["class_id"] = int(values[4])
                    else:
                        annotation["confidence"] = float(values[4])
                    annotations.append(annotation)
            yield relative, annotations


def lap_group(relative: str) -> str:
    name = Path(relative).name
    match = LAP_GROUP_RE.match(name)
    return match.group(1) if match else Path(name).stem


def anime256_group(filename: str) -> str:
    # Filenames generally end in a unique crop index. Dropping that suffix keeps
    # multiple faces from the same source image in the same split.
    stem = Path(filename).stem
    head, sep, tail = stem.rpartition("-")
    return head if sep and tail.isdigit() else stem


def build_lap(raw: Path, normalized: Path):
    root = raw / "lap" / "anime"
    classes = {}
    with (root / "classes.txt").open(encoding="utf-8-sig") as stream:
        for line in stream:
            class_id, name = line.strip().split(":", 1)
            classes[int(class_id)] = name
    (normalized / "lap_classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manual = dict(parse_lap_labels(root / "labels_faces.txt", "manual"))
    baseline = dict(parse_lap_labels(root / "labels_faces_baseline.txt", "baseline"))
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    image_rows, manual_rows, baseline_rows, errors = [], [], [], []
    for path in images:
        relative = path.relative_to(root).as_posix()
        width, height = image_size(path)
        group = lap_group(relative)
        split = stable_split("lap", group)
        image_id = f"lap:{relative}"
        common = {
            "image_id": image_id,
            "image_path": f"raw/lap/anime/{relative}",
            "source_dataset": "lap",
            "source_group": group,
            "split": split,
            "width": width,
            "height": height,
        }
        image_rows.append({
            **common,
            "manual_face_labels": int(relative in manual),
            "baseline_face_labels": int(relative in baseline),
        })
        for label_kind, source, sink in (
            ("manual_refined", manual, manual_rows),
            ("automatic_baseline", baseline, baseline_rows),
        ):
            if relative not in source:
                continue
            annotations = source[relative]
            for annotation in annotations:
                x1, y1, x2, y2 = annotation["bbox"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{label_kind}:{relative}:{annotation['bbox']} outside {width}x{height}")
            sink.append({**common, "annotation_source": label_kind, "annotations": annotations})
    dump_jsonl(normalized / "lap_faces_manual.jsonl", manual_rows)
    dump_jsonl(normalized / "lap_faces_baseline.jsonl", baseline_rows)
    return image_rows, errors


def build_manga109(raw: Path, normalized: Path):
    root = raw / "manga109"
    image_rows, records, errors = [], [], []
    for xml_path in sorted((root / "annotations").glob("*.xml")):
        tree = ET.parse(xml_path)
        book = tree.getroot()
        title = book.attrib["title"]
        split = stable_split("manga109", title)
        pages = book.find("pages")
        if pages is None:
            errors.append(f"missing <pages> element: {xml_path.name}")
            continue
        for page in pages:
            index = int(page.attrib["index"])
            width, height = int(page.attrib["width"]), int(page.attrib["height"])
            relative = f"{title}/{index:03d}.jpg"
            image_path = root / "images" / relative
            image_id = f"manga109:{title}:{index:03d}"
            annotations = []
            for face in page.findall("face"):
                bbox = [int(face.attrib[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                x1, y1, x2, y2 = bbox
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{relative}:{bbox} outside {width}x{height}")
                annotations.append({
                    "bbox": bbox,
                    "category": "face",
                    "character_id": face.attrib.get("character"),
                    "source_id": face.attrib.get("id"),
                })
            if not image_path.is_file():
                errors.append(f"missing image: {relative}")
                continue
            common = {
                "image_id": image_id,
                "image_path": f"raw/manga109/images/{relative}",
                "source_dataset": "manga109",
                "source_group": title,
                "split": split,
                "width": width,
                "height": height,
            }
            image_rows.append({**common, "manual_face_labels": 1, "baseline_face_labels": 0})
            records.append({**common, "annotation_source": "manga109_v2026", "annotations": annotations})
    dump_jsonl(normalized / "manga109_faces.jsonl", records)
    return image_rows, errors


def build_anime256(raw: Path):
    root = raw / "anime256"
    image_rows, errors = [], []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        relative = path.relative_to(root).as_posix()
        width, height = image_size(path)
        if (width, height) != (256, 256):
            errors.append(f"unexpected size: {relative}: {width}x{height}")
        group = anime256_group(path.name)
        image_rows.append({
            "image_id": f"anime256:{relative}",
            "image_path": f"raw/anime256/{relative}",
            "source_dataset": "anime256",
            "source_group": group,
            "split": stable_split("anime256", group),
            "width": width,
            "height": height,
            "manual_face_labels": 0,
            "baseline_face_labels": 0,
        })
    return image_rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.datasets_root.resolve()
    raw, manifests = root / "raw", root / "manifests"
    normalized = manifests / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)

    all_rows, all_errors = [], {}
    for name, builder in (
        ("lap", lambda: build_lap(raw, normalized)),
        ("manga109", lambda: build_manga109(raw, normalized)),
        ("anime256", lambda: build_anime256(raw)),
    ):
        rows, errors = builder()
        all_rows.extend(rows)
        all_errors[name] = errors

    fields = [
        "image_id", "image_path", "source_dataset", "source_group", "split",
        "width", "height", "manual_face_labels", "baseline_face_labels",
    ]
    with (manifests / "images.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    groups = {}
    for row in all_rows:
        key = (row["source_dataset"], row["source_group"])
        groups[key] = row["split"]
    with (manifests / "groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_dataset", "source_group", "split"])
        for (dataset, group), split in sorted(groups.items()):
            writer.writerow([dataset, group, split])

    counts = Counter((row["source_dataset"], row["split"]) for row in all_rows)
    summary = {
        "schema_version": "1.0",
        "split_seed": SPLIT_SEED,
        "split_policy": "sha256(dataset, source_group), 75/10/15 thresholds",
        "images": {
            dataset: {
                split: counts[(dataset, split)] for split in ("train", "validation", "test")
            }
            for dataset in ("lap", "manga109", "anime256")
        },
        "validation_error_counts": {name: len(errors) for name, errors in all_errors.items()},
        "normalized_annotations": {
            "lap_manual": jsonl_stats(normalized / "lap_faces_manual.jsonl"),
            "lap_baseline": jsonl_stats(normalized / "lap_faces_baseline.jsonl"),
            "manga109_v2026": jsonl_stats(normalized / "manga109_faces.jsonl"),
        },
    }
    summary["groups"] = {
        dataset: sum(1 for ds, _ in groups if ds == dataset)
        for dataset in ("lap", "manga109", "anime256")
    }
    (manifests / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (manifests / "validation_errors.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for dataset, errors in all_errors.items():
            for error in errors:
                stream.write(f"{dataset}: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
