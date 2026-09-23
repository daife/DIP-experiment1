#!/usr/bin/env python3
"""Apply human crop-review decisions to an existing step-three dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_detection_dataset import ROOT, load_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    parser.add_argument("--review-csv", type=Path, default=ROOT / "datasets/annotations/corrected/step3_crop_review.csv")
    args = parser.parse_args()
    review = load_review(args.review_csv)
    if not review:
        raise ValueError(f"No review decisions found in {args.review_csv}")
    rows = [json.loads(line) for line in (args.dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    base_ids = {row["sample_id"] for row in rows if row["augmentation"] is None}
    unknown = set(review) - base_ids
    if unknown:
        raise ValueError(f"Unknown sample IDs: {sorted(unknown)[:10]}")
    counts = Counter()
    usable = []
    for row in rows:
        base_id = row.get("parent_sample_id", row["sample_id"])
        decision = review.get(base_id, {"decision": "", "tags": [], "note": ""})
        row["review_decision"] = decision["decision"] or "unreviewed"
        row["review_tags"] = decision["tags"]
        row["review_note"] = decision["note"]
        if row["augmentation"] is None:
            counts[row["review_decision"]] += 1
        if row["review_decision"] != "reject":
            usable.append(row)
    # The usable manifest is an explicit gate for later training. Unreviewed
    # rows remain marked, so consumers can require a fully reviewed dataset.
    with (args.dataset / "usable_samples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in usable:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"base_decisions": dict(counts), "usable_crops_including_augmentations": len(usable),
               "fully_reviewed": counts["unreviewed"] == 0}
    (args.dataset / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
