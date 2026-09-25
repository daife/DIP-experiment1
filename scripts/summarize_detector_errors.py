#!/usr/bin/env python3
"""Summarize saved detector errors without rescanning licensed pages."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest_path = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    stage_path = ROOT / "results/step4_stage_stats.json"
    review_path = ROOT / "datasets/annotations/corrected/step4_hard_negative_review.csv"
    manifest = {row["image_id"]: row for row in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines())}
    stages = read_json(stage_path)
    with review_path.open(newline="", encoding="utf-8-sig") as stream:
        reviewed = list(csv.DictReader(stream))
    output = {
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (manifest_path, stage_path, review_path)},
        "crop_test": {"negative_total": stages["test"][0]["negative_entering"],
                      "negative_passed_all_stages": stages["test"][-1]["negative_passing"],
                      "positive_total": stages["test"][0]["positive_entering"],
                      "positive_passed_all_stages": stages["test"][-1]["positive_passing"]},
        "hard_negative_review": {"reviewed": len(reviewed),
                                 "approved_negative": sum(row["decision"] == "keep" for row in reviewed)},
        "pages": {},
    }
    for split in ("validation", "test"):
        path = ROOT / f"results/step8_candidate_verifier_{split}.json"
        data = read_json(path)
        output["inputs_sha256"][str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        dimensions = []
        for page in data["pages"]:
            for annotation in manifest[page["id"]]["annotations"]:
                x1, y1, x2, y2 = annotation["bbox"]
                dimensions.append((x2 - x1, y2 - y1))
        ratios = [max(w / h, h / w) for w, h in dimensions]
        baseline, filtered = data["summary"]["baseline"], data["summary"]["hog0.75_side36"]
        output["pages"][split] = {
            "source_groups": len(data["pages"]), "annotated_faces": len(dimensions),
            "aspect_ratio_gt_2_25": sum(ratio > 2.25 for ratio in ratios),
            "max_side_lt_36": sum(max(w, h) < 36 for w, h in dimensions),
            "cascade": baseline, "current_demo": filtered,
            "tp_count_difference": baseline["tp"] - filtered["tp"],
            "fp_count_difference": baseline["fp"] - filtered["fp"],
        }
    target = ROOT / "results/step8_error_diagnosis.json"
    target.write_bytes((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(target), "summary": output["pages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
