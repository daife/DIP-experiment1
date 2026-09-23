#!/usr/bin/env python3
"""Check step-three crop provenance, splits, negative boxes, and channel cache."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_detection_dataset import ROOT, near_face


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/normalized/manga109_faces.jsonl")
    args = parser.parse_args()
    sources = {record["image_id"]: record for record in
               (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines())}
    rows = [json.loads(line) for line in (args.dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [row["sample_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate sample ID"
    group_splits = defaultdict(set)
    counts = Counter()
    base = {}
    for row in rows:
        source = sources[row["parent_image_id"]]
        assert row["split"] == source["split"]
        assert row["source_group"] == source["source_group"]
        assert row["parent_image_path"] == source["image_path"]
        group_splits[row["source_group"]].add(row["split"])
        with Image.open(args.dataset / row["file"]) as image:
            assert image.size == (24, 24) and image.mode == "L"
        counts[(row["split"], row["label"], row["augmentation"] is not None)] += 1
        if row["augmentation"] is None:
            base[row["sample_id"]] = row
            if row["label"] == 0:
                assert not near_face(row["crop_bbox_xyxy"], [ann["bbox"] for ann in source["annotations"]]), row["sample_id"]
            else:
                assert row["source_bbox_xyxy"] == source["annotations"][row["annotation_index"]]["bbox"]
        else:
            assert row["split"] == "train" and row["augmentation"] == "horizontal_flip"
            assert row["parent_sample_id"] in base
    assert all(len(splits) == 1 for splits in group_splits.values()), "source group crosses splits"
    channels = np.load(args.dataset / "channels11.npy", mmap_mode="r")
    assert channels.shape == (len(rows), 11, 24, 24) and channels.dtype == np.uint8
    for index in (0, len(rows) // 2, len(rows) - 1):
        with Image.open(args.dataset / rows[index]["file"]) as image:
            assert np.array_equal(channels[index, 0], np.asarray(image))
    usable_path = args.dataset / "usable_samples.jsonl"
    if usable_path.exists():
        usable = [json.loads(line) for line in usable_path.read_text(encoding="utf-8").splitlines()]
        with (args.dataset / "usable_channels11_index.csv").open(encoding="utf-8", newline="") as stream:
            usable_index = list(csv.DictReader(stream))
        assert len(usable) == len(usable_index)
        for row, mapped in zip(usable, usable_index):
            assert row["sample_id"] == mapped["sample_id"]
            assert rows[int(mapped["array_index"])]["sample_id"] == row["sample_id"]
            assert row["review_decision"] != "reject"
        summary = json.loads((args.dataset / "review_summary.json").read_text(encoding="utf-8"))
        assert len(usable) == summary["usable_crops_including_augmentations"]
    print(json.dumps({"crops": len(rows), "base_crops": len(base), "source_groups": len(group_splits),
                      "counts": {str(k): v for k, v in sorted(counts.items())}}, indent=2))


if __name__ == "__main__":
    main()
