"""Select a fixed 320-image review cohort and assemble its preannotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/manifests/images.csv"
PILOT = ROOT / "datasets/annotations/auto/hrnetv2_cpu_pilot_64/annotations.jsonl"
NEW = ROOT / "datasets/annotations/auto/hrnetv2_review_256_v2/annotations.jsonl"
OUTPUT = ROOT / "datasets/annotations/review_sets/landmark320"
TRACKED_IMAGE_DIR = OUTPUT / "images"
SEED = 20260925
TARGET = {"train": 240, "validation": 32, "test": 48}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> tuple[list[str], list[dict[str, str]]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [row for row in reader if row["source_dataset"] == "anime256"]


def select() -> None:
    fields, pool = load_manifest()
    pilot = read_jsonl(PILOT)
    pilot_ids = {row["image_id"] for row in pilot}
    if len(pilot) != 64 or len(pilot_ids) != 64:
        raise ValueError("Expected the intact 64-image pilot")
    used_groups = {row["source_group"] for row in pilot}
    pilot_counts = Counter(row["split"] for row in pilot)
    picked: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in pool:
            if row["split"] == split and row["source_group"] not in used_groups and row["image_id"] not in pilot_ids:
                by_group[row["source_group"]].append(row)
        groups = sorted(by_group)
        rng = random.Random(SEED + {"train": 0, "validation": 1, "test": 2}[split])
        rng.shuffle(groups)
        need = TARGET[split] - pilot_counts[split]
        if need < 0 or len(groups) < need:
            raise ValueError(f"Insufficient distinct groups in {split}")
        for group in groups[:need]:
            picked.append(rng.choice(sorted(by_group[group], key=lambda r: r["image_id"])))
            used_groups.add(group)
    counts = Counter(row["split"] for row in picked)
    if len(picked) != 256 or any(counts[s] + pilot_counts[s] != TARGET[s] for s in TARGET):
        raise AssertionError("Selection quotas not met")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "supplement_selection.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(picked)
    print(json.dumps({"pilot": dict(pilot_counts), "supplement": dict(counts), "selected": len(picked), "path": str(path.relative_to(ROOT))}, ensure_ascii=False))


def assemble() -> None:
    pilot = read_jsonl(PILOT)
    supplement = read_jsonl(NEW)
    selection_path = OUTPUT / "supplement_selection.csv"
    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        selected_ids = {row["image_id"] for row in csv.DictReader(handle)}
    if len(supplement) != 256 or {row["image_id"] for row in supplement} != selected_ids:
        raise ValueError("Supplement annotation IDs do not match fixed selection")
    combined = pilot + supplement
    ids = [row["image_id"] for row in combined]
    counts = Counter(row["split"] for row in combined)
    if len(combined) != 320 or len(set(ids)) != 320 or dict(counts) != TARGET:
        raise ValueError("Combined review cohort is not 240/32/48 unique images")
    group_splits: dict[str, set[str]] = defaultdict(set)
    images = []
    for record in combined:
        group_splits[record["source_group"]].add(record["split"])
        points = record["annotations"][0]["landmarks"]
        if len(points) != 28 or any(point["visibility"] is not None for point in points):
            raise ValueError(f"Unexpected landmark state: {record['image_id']}")
        filename = Path(record["image_path"]).name
        tracked_path = TRACKED_IMAGE_DIR / filename
        path = tracked_path if tracked_path.is_file() else ROOT / "datasets" / record["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        image_path = str(path.relative_to(ROOT / "datasets")).replace("\\", "/")
        record["image_path"] = image_path
        images.append({
            "image_id": record["image_id"],
            "image_path": image_path,
            "split": record["split"],
            "source_group": record["source_group"],
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A source group crosses splits")
    annotation_path = OUTPUT / "annotations.jsonl"
    with annotation_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in combined:
            record["landmark_schema_id"] = "anime_face_detector_hrnetv2_28_v1"
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    pilot_run_metadata = PILOT.with_name("run_metadata.json")
    supplement_run_metadata = NEW.with_name("run_metadata.json")
    if not pilot_run_metadata.is_file() or not supplement_run_metadata.is_file():
        raise FileNotFoundError("Both preannotation runs must have metadata")
    (OUTPUT / "pilot_run_metadata.json").write_bytes(pilot_run_metadata.read_bytes().replace(b"\r\n", b"\n"))
    (OUTPUT / "supplement_run_metadata.json").write_bytes(supplement_run_metadata.read_bytes().replace(b"\r\n", b"\n"))
    metadata = {
        "schema_version": "1.0",
        "cohort": "landmark320",
        "counts": TARGET,
        "selection_seed": SEED,
        "split_source": "datasets/manifests/images.csv; fixed source_group split",
        "pilot_annotations": str(PILOT.relative_to(ROOT)).replace("\\", "/"),
        "pilot_sha256": sha256(PILOT),
        "supplement_annotations": str(NEW.relative_to(ROOT)).replace("\\", "/"),
        "supplement_sha256": sha256(NEW),
        "pilot_run_metadata_sha256": sha256(OUTPUT / "pilot_run_metadata.json"),
        "supplement_run_metadata_sha256": sha256(OUTPUT / "supplement_run_metadata.json"),
        "supplement_selection_sha256": sha256(selection_path),
        "annotations_sha256": sha256(annotation_path),
        "archive_url": "https://github.com/luzhixing12345/anime-face-dataset/releases/download/v0.0.1/anime256.zip",
        "archive_sha256": "5e266c8954dc75f0d0bbf01e537e0097faba34c3cd7d7efbc7195bfabec07297",
        "image_count": len(images),
        "image_bytes": sum(image["bytes"] for image in images),
        "images": images,
    }
    with (OUTPUT / "review_set.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "images": len(images), "bytes": metadata["image_bytes"], "annotations_sha256": metadata["annotations_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "assemble"))
    args = parser.parse_args()
    {"select": select, "assemble": assemble}[args.command]()
