#!/usr/bin/env python3
"""Compare baseline and fixed HOG gates on distinct validation works."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.detection_metrics import match_detections
from src.multiscale import PyramidConfig, detect_multiscale, nms
from src.candidate_verifier import CandidateVerifier, features
from compare_multiscale import choose_pages


def choose_test_pages(manifest: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == "test" and row["annotations"]]
    rows.sort(key=lambda row: hashlib.sha256(("step8-test:" + row["image_id"]).encode()).digest())
    chosen, groups = [], set()
    for row in rows:
        if row["source_group"] not in groups:
            chosen.append(row)
            groups.add(row["source_group"])
            if len(chosen) == count:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--verifier", type=Path, default=ROOT / "models/candidate_hog_svm.npz")
    parser.add_argument("--cascade", type=Path, default=ROOT / "models/step4_cascade.json")
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--post-filter-nms", type=float,
                        help="second NMS after HOG/side filtering")
    parser.add_argument("--post-filter-score", choices=("cascade", "verifier"), default="cascade")
    parser.add_argument("--output", type=Path)
    parser.add_argument('--cache-dir', type=Path, help='Local model-independent Cascade/HOG cache; timings include cache reads when reused')
    args = parser.parse_args()
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        state = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'src').glob('*.py')}
        state_path = args.cache_dir / 'source_state.json'
        if state_path.exists():
            if json.loads(state_path.read_text()) != state:
                parser.error('source changed since cache creation; use a new cache directory')
        elif any(args.cache_dir.glob('*.npz')):
            parser.error('cache lacks source provenance; use a new cache directory')
        else:
            state_path.write_text(json.dumps(state, indent=2)+'\n')
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    cascade_path = args.cascade
    verifier_path = args.verifier
    cascade = cascade_from_dict(json.loads(cascade_path.read_text(encoding="utf-8")))
    verifier = CandidateVerifier(verifier_path)
    rules = ({"baseline": (None, None), "rbf1.5_side36": (1.5, 36)}
             if verifier_path.suffix == ".joblib" and args.split == "test" else
             {"baseline": (None, None), **{f"rbf{threshold:g}_side36": (threshold, 36)
                                            for threshold in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)}}
             if verifier_path.suffix == ".joblib" else
             {"baseline": (None, None), "hog0.5_side48": (0.5, 48),
              "hog0.75_side48": (0.75, 48), "hog0.75_side36": (0.75, 36),
              "hog0_side80": (0.0, 80)} if args.split == "validation" else
             {"baseline": (None, None), "hog0.75_side36": (0.75, 36)})
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in rules}
    pages = []
    if args.pages < 1:
        parser.error("--pages must be positive")
    selected = choose_pages(manifest, args.pages) if args.split == "validation" else choose_test_pages(manifest, args.pages)
    for page in selected:
        page_started = perf_counter()
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        gt = np.asarray([a["bbox"] for a in page["annotations"]], dtype=np.int32).reshape(-1, 4)
        cache_path = None
        if args.cache_dir:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            identity = ':'.join((page['image_id'], hashlib.sha256(cascade_path.read_bytes()).hexdigest(),
                                 hashlib.sha256(manifest.read_bytes()).hexdigest(), str(args.nms_iou),
                                 hashlib.sha256((ROOT / 'src/candidate_verifier.py').read_bytes()).hexdigest(),
                                 hashlib.sha256((ROOT / 'src/multiscale.py').read_bytes()).hexdigest(),
                                 hashlib.sha256((ROOT / 'datasets' / page['image_path']).read_bytes()).hexdigest()))
            cache_path = args.cache_dir / (hashlib.sha256(identity.encode()).hexdigest()+'.npz')
        reused = bool(cache_path and cache_path.exists())
        if reused:
            with np.load(cache_path) as data:
                boxes, scores, vectors = data['boxes'], data['scores'], data['features']
        else:
            boxes, scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, args.nms_iou))
        scan_seconds = perf_counter() - page_started
        verifier_started = perf_counter()
        if not reused:
            vectors = features(gray, boxes)
            if cache_path:
                np.savez_compressed(cache_path, boxes=boxes, scores=scores, features=vectors)
        hog_scores = (verifier.model.decision_function(vectors) if len(vectors) else np.empty(0)) if verifier.model is not None else vectors @ verifier.coefficients + verifier.intercept
        verifier_seconds = perf_counter() - verifier_started
        item = {"id": page["image_id"], "gt": len(gt), "candidates": len(boxes),
                "cache_reused": reused,
                "scan_seconds": scan_seconds, "verifier_seconds": verifier_seconds, "rules": {}}
        for name, (threshold, side) in rules.items():
            keep = np.ones(len(boxes), dtype=bool) if threshold is None else ((hog_scores >= threshold) & (boxes[:, 2] - boxes[:, 0] >= side))
            selected_boxes, selected_scores = boxes[keep], scores[keep]
            if threshold is not None and args.post_filter_nms is not None and len(selected_boxes):
                order_scores = selected_scores if args.post_filter_score == "cascade" else hog_scores[keep]
                indices = nms(selected_boxes, order_scores, args.post_filter_nms)
                selected_boxes, selected_scores = selected_boxes[indices], selected_scores[indices]
            result = match_detections(selected_boxes, selected_scores, gt, 0.5)
            item["rules"][name] = {k: result[k] for k in ("tp", "fp", "fn")}
            for key in ("tp", "fp", "fn"):
                totals[name][key] += result[key]
        pages.append(item)
        item["total_seconds"] = perf_counter() - page_started
        print(item, flush=True)
    for result in totals.values():
        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        result["precision"] = tp / (tp + fp) if tp + fp else 0
        result["recall"] = tp / (tp + fn) if tp + fn else 0
        result["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    output = {"split": args.split, "step": 2, "scale_factor": 1.2, "nms_iou": args.nms_iou,
              "post_filter_nms": args.post_filter_nms,
              "post_filter_score": args.post_filter_score,
              "timing_scope": "includes cache IO and feature computation; cache reuse bypasses Cascade/HOG, not deployment latency" if args.cache_dir else "ordinary scan and verifier; no cache",
              "cascade_sha256": hashlib.sha256(cascade_path.read_bytes()).hexdigest(),
              "verifier_sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "selection": ("step5-comparison" if args.split == "validation" else "step8-test") + f" SHA-256 order, {args.pages} distinct source groups",
              "pages": pages, "summary": totals,
              "mean_scan_seconds": sum(p["scan_seconds"] for p in pages) / len(pages),
              "mean_verifier_seconds": sum(p["verifier_seconds"] for p in pages) / len(pages),
              "mean_total_seconds": sum(p["total_seconds"] for p in pages) / len(pages)}
    target = args.output or ROOT / f"results/step8_candidate_verifier_{args.split}.json"
    target.write_bytes((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(totals)


if __name__ == "__main__":
    main()
