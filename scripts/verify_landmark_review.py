"""Validate the fixed 320-image coarse human-accepted landmark review."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "datasets/annotations/review_sets/landmark320"
CORRECTED = ROOT / "datasets/annotations/corrected"
ASSISTED = COHORT / "assisted_review_v2.jsonl"
ACCEPTED = CORRECTED / "landmark28_review320.jsonl"
ATTESTATION = CORRECTED / "landmark28_review320_acceptance.json"
OUTPUT = ROOT / "results/landmark_review320_coarse_stats.json"
EXPECTED_SPLITS = {"train": 240, "validation": 32, "test": 48}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    attestation = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    summary = json.loads((COHORT / "assisted_review_v2_summary.json").read_text(encoding="utf-8"))
    original = {r["image_id"]: r for r in read_jsonl(COHORT / "annotations.jsonl")}
    assisted = {r["image_id"]: r for r in read_jsonl(ASSISTED)}
    accepted = read_jsonl(ACCEPTED)
    if len(original) != 320 or len(assisted) != 320 or len(accepted) != 320:
        raise ValueError("Review cohort must contain 320 unique records in each layer")
    if len({r["image_id"] for r in accepted}) != 320 or {r["image_id"] for r in accepted} != set(original):
        raise ValueError("Accepted image IDs differ from the fixed cohort")
    if sha256(ASSISTED) != summary["output_sha256"] or sha256(ASSISTED) != attestation["assisted_review_sha256"]:
        raise ValueError("Assisted review hash differs from summary or attestation")
    if sha256(COHORT / "annotations.jsonl") != summary["input_sha256"]:
        raise ValueError("Fixed preannotation hash differs from assisted summary")

    splits: Counter[str] = Counter()
    types: Counter[str] = Counter()
    fully_unreliable = []
    changed_coordinates = 0
    for record in accepted:
        image_id = record["image_id"]
        source = original[image_id]
        prior = assisted[image_id]
        if (record["split"], record["source_group"], record["image_path"]) != (
            source["split"], source["source_group"], source["image_path"]
        ):
            raise ValueError(f"Source identity changed: {image_id}")
        if record["review"]["status"] != "human_coarse_review_accepted" or record["review"]["coordinate_refinement"] is not False:
            raise ValueError(f"Review level differs from user attestation: {image_id}")
        if record["assisted_review"] != prior["review"]:
            raise ValueError(f"Assisted provenance changed: {image_id}")
        splits[record["split"]] += 1
        points = record["annotations"][0]["landmarks"]
        old_points = source["annotations"][0]["landmarks"]
        if len(points) != len(old_points) or len(points) != 28:
            raise ValueError(f"Point count differs: {image_id}")
        if all(point["review_type"] == "U" for point in points):
            fully_unreliable.append({"image_id": image_id, "split": record["split"]})
        for point, old in zip(points, old_points, strict=True):
            expected = {"V": (1, None), "H": (0, "reliable_estimate"), "U": (0, "uncertain")}
            kind = point.get("review_type")
            if kind not in expected or (point["visibility"], point.get("coordinate_quality")) != expected[kind]:
                raise ValueError(f"Invalid visibility or coordinate quality: {image_id}")
            if point["confidence"] != old["confidence"]:
                raise ValueError(f"Model confidence changed: {image_id}")
            changed_coordinates += (point["x"], point["y"]) != (old["x"], old["y"])
            types[kind] += 1
    if dict(splits) != EXPECTED_SPLITS or changed_coordinates != 0 or sum(types.values()) != 8960:
        raise ValueError("Unexpected split, point count, or coordinate changes")
    result = {
        "status": "PASS",
        "review_level": attestation["review_level"],
        "records": len(accepted),
        "split_counts": EXPECTED_SPLITS,
        "point_counts": dict(types),
        "fully_unreliable_images": fully_unreliable,
        "coordinate_changes_from_preannotation": changed_coordinates,
        "assisted_sha256": sha256(ASSISTED),
        "accepted_sha256": sha256(ACCEPTED),
        "limitations": "Human accepted assisted V/H/U labels after simple visual review; no point coordinates were fine-tuned. Do not call coordinates precise human ground truth.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
