#!/usr/bin/env python3
"""Train a depth-2 Cascade from fully reviewed step-three crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade import cascade_to_dict, fit_cascade, stage_statistics  # noqa: E402


def load_reviewed_data(dataset: Path) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict]]]:
    summary = json.loads((dataset / "review_summary.json").read_text(encoding="utf-8"))
    if summary.get("fully_reviewed") is not True:
        raise ValueError("step-three crop review is incomplete")
    rows = [json.loads(line) for line in (dataset / "usable_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    with (dataset / "usable_channels11_index.csv").open(newline="", encoding="utf-8") as stream:
        indices = list(csv.DictReader(stream))
    if len(rows) != len(indices) or any(row["sample_id"] != item["sample_id"] for row, item in zip(rows, indices, strict=True)):
        raise ValueError("reviewed manifest and channel index do not align")
    if any(row["review_decision"] != "keep" for row in rows):
        raise ValueError("usable manifest contains a sample that was not approved")
    cache = np.load(dataset / "channels11.npy", mmap_mode="r")
    if cache.ndim != 4 or cache.shape[1:] != (11, 24, 24) or cache.dtype != np.uint8:
        raise ValueError("invalid 11-channel cache")
    array_indices = np.array([int(item["array_index"]) for item in indices], dtype=np.intp)
    if np.any(array_indices < 0) or np.any(array_indices >= len(cache)) or len(set(array_indices)) != len(array_indices):
        raise ValueError("invalid or duplicate channel cache index")
    result = {}
    for split in ("train", "validation", "test"):
        selected = [i for i, row in enumerate(rows) if row["split"] == split]
        result[split] = (
            np.asarray(cache[array_indices[selected]]),
            np.array([rows[i]["label"] for i in selected], dtype=np.uint8),
            [rows[i] for i in selected],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    parser.add_argument("--model", type=Path, default=ROOT / "models/step4_initial_cascade.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/step4_initial_stage_stats.json")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--trees-per-stage", type=int, default=6)
    parser.add_argument("--root-candidates", type=int, default=16)
    parser.add_argument("--child-candidates", type=int, default=16)
    parser.add_argument("--target-stage-recall", type=float, default=0.99)
    parser.add_argument("--hard-negative-dir", type=Path)
    parser.add_argument("--hard-review-csv", type=Path)
    args = parser.parse_args()
    data = load_reviewed_data(args.dataset)
    train_x, train_y, _ = data["train"]
    validation_x, validation_y, _ = data["validation"]
    hard_ids = []
    if args.hard_negative_dir is not None:
        if args.hard_review_csv is None:
            raise ValueError("hard negatives require --hard-review-csv with explicit keep decisions")
        with args.hard_review_csv.open(newline="", encoding="utf-8-sig") as stream:
            review = list(csv.DictReader(stream))
        if not review or set(review[0]) != {"id", "decision", "reason"}:
            raise ValueError("hard review CSV needs id,decision,reason columns")
        if len({row["id"] for row in review}) != len(review):
            raise ValueError("duplicate hard-negative review ID")
        keep = {row["id"] for row in review if row["decision"] == "keep"}
        if any(row["decision"] not in ("keep", "reject") for row in review):
            raise ValueError("hard-negative decision must be keep or reject")
        candidates = [json.loads(line) for line in (args.hard_negative_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        lookup = {row["id"]: index for index, row in enumerate(candidates)}
        if not keep or not keep <= lookup.keys():
            raise ValueError("no valid approved hard-negative IDs")
        if any(candidates[lookup[item]]["split"] != "train" or candidates[lookup[item]]["label"] != 0 for item in keep):
            raise ValueError("hard-negative source must be train and label 0")
        channel_cache = np.load(args.hard_negative_dir / "channels11.npy", mmap_mode="r")
        if channel_cache.shape != (len(candidates), 11, 24, 24) or channel_cache.dtype != np.uint8:
            raise ValueError("hard-negative cache does not match manifest")
        hard_ids = sorted(keep)
        hard_windows = np.asarray(channel_cache[[lookup[item] for item in hard_ids]])
        train_x = np.concatenate((train_x, hard_windows), axis=0)
        train_y = np.concatenate((train_y, np.zeros(len(hard_ids), dtype=np.uint8)))
    cascade = fit_cascade(
        train_x, train_y, validation_x, validation_y,
        seed=args.seed, num_stages=args.stages, num_trees=args.trees_per_stage,
        root_candidates=args.root_candidates, child_candidates=args.child_candidates,
        target_recall=args.target_stage_recall,
    )
    payload = cascade_to_dict(cascade)
    payload["training"] = {
        "seed": args.seed, "stages": args.stages, "trees_per_stage": args.trees_per_stage,
        "root_candidates": args.root_candidates, "child_candidates": args.child_candidates,
        "target_stage_recall": args.target_stage_recall,
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "fully_reviewed": True,
        "approved_hard_negative_ids": hard_ids,
        "hard_negative_dir": str(args.hard_negative_dir) if hard_ids else None,
    }
    report = {split: stage_statistics(cascade, windows, labels) for split, (windows, labels, _) in data.items()}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(args.model), "report": str(args.report), "statistics": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
