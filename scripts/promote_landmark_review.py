"""Preserve assisted labels and add an explicit coarse human-acceptance layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "datasets/annotations/review_sets/landmark320"
SOURCE = COHORT / "assisted_review_v2.jsonl"
BASELINE = COHORT / "annotations.jsonl"
ATTESTATION = ROOT / "datasets/annotations/corrected/landmark28_review320_acceptance.json"
OUTPUT = ROOT / "datasets/annotations/corrected/landmark28_review320.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    attestation = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    if sha256(SOURCE) != attestation["assisted_review_sha256"]:
        raise ValueError("Assisted review differs from the accepted source hash")
    if attestation["coordinate_refinement"] is not False or attestation["cohort"] != "landmark320":
        raise ValueError("Expected a coarse acceptance of the fixed landmark320 cohort")
    assisted = read_jsonl(SOURCE)
    baseline = {row["image_id"]: row for row in read_jsonl(BASELINE)}
    if len(assisted) != 320 or len(baseline) != 320 or {r["image_id"] for r in assisted} != set(baseline):
        raise ValueError("Review does not cover the exact 320-image cohort")
    counts = {"train": 0, "validation": 0, "test": 0}
    output_rows = []
    for row in assisted:
        original = baseline[row["image_id"]]
        if row["split"] != original["split"] or row["source_group"] != original["source_group"]:
            raise ValueError(f"Split or group mismatch: {row['image_id']}")
        points = row["annotations"][0]["landmarks"]
        original_points = original["annotations"][0]["landmarks"]
        if len(points) != 28 or len(original_points) != 28:
            raise ValueError(f"Point count mismatch: {row['image_id']}")
        for point, source_point in zip(points, original_points, strict=True):
            if (point["x"], point["y"]) != (source_point["x"], source_point["y"]):
                raise ValueError("Coordinates changed despite no-refinement attestation")
            if point["confidence"] != source_point["confidence"]:
                raise ValueError("Model confidence changed")
            if (point.get("review_type"), point["visibility"], point.get("coordinate_quality")) not in (
                ("V", 1, None), ("H", 0, "reliable_estimate"), ("U", 0, "uncertain")
            ):
                raise ValueError(f"Invalid V/H/U assignment: {row['image_id']}")
        assisted_review = row.pop("review")
        if assisted_review.get("reviewer_claim") != "not_human_reviewed":
            raise ValueError("Unexpected assisted-review provenance")
        row["assisted_review"] = assisted_review
        row["review"] = {
            "status": "human_coarse_review_accepted",
            "method": "user_full_cohort_visual_acceptance_of_assisted_labels",
            "accepted_at_utc": attestation["recorded_at_utc"],
            "attestation_file": str(ATTESTATION.relative_to(ROOT)).replace("\\", "/"),
            "assisted_review_sha256": attestation["assisted_review_sha256"],
            "coordinate_refinement": False,
            "coordinate_policy": "original_hrnetv2_predictions_unchanged",
            "scope": "all_320_images",
        }
        counts[row["split"]] += 1
        output_rows.append(row)
    if counts != {"train": 240, "validation": 32, "test": 48}:
        raise ValueError(f"Unexpected split counts: {counts}")
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output_rows).encode("utf-8")
    if OUTPUT.is_file() and OUTPUT.read_bytes() != content:
        raise ValueError("Existing corrected file differs; refusing to overwrite later human edits")
    OUTPUT.write_bytes(content)
    print(f"Wrote {len(output_rows)} coarse human-accepted records to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
