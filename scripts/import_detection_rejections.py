#!/usr/bin/env python3
"""Convert a reviewed reject-only CSV into the canonical full crop review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.prepare_detection_dataset import ROOT, REVIEW_FIELDS
except ModuleNotFoundError:
    from prepare_detection_dataset import ROOT, REVIEW_FIELDS


def import_rejections(source: Path, samples: Path, output: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
    base_ids = [row["sample_id"] for row in rows if row["augmentation"] is None]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("Duplicate base sample IDs")
    rejected: dict[str, tuple[str, str]] = {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for number, fields in enumerate(csv.reader(stream), 1):
            if len(fields) != len(REVIEW_FIELDS):
                raise ValueError(f"{source}:{number}: expected four CSV columns")
            sample_id, decision, tags, note = (field.strip() for field in fields)
            if decision.lower() != "reject":
                raise ValueError(f"{source}:{number}: expected reject decision")
            if sample_id in rejected:
                raise ValueError(f"{source}:{number}: duplicate ID {sample_id}")
            rejected[sample_id] = (tags, note)
    unknown = set(rejected) - set(base_ids)
    if unknown:
        raise ValueError(f"Unknown IDs: {sorted(unknown)[:10]}")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(REVIEW_FIELDS)
        for sample_id in base_ids:
            if sample_id in rejected:
                tags, note = rejected[sample_id]
                writer.writerow([sample_id, "reject", tags, note])
            else:
                writer.writerow([sample_id, "keep", "", ""])
    return {"base_samples": len(base_ids), "keep": len(base_ids) - len(rejected), "reject": len(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/annotations/corrected/revise_reason.csv")
    parser.add_argument("--samples", type=Path, default=ROOT / "datasets/derived/step3_detection_v1/samples.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/annotations/corrected/step3_crop_review.csv")
    args = parser.parse_args()
    print(json.dumps(import_rejections(args.source, args.samples, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
