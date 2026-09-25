#!/usr/bin/env python3
"""Compare integer channels with float32 arithmetic under identical rounding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict  # noqa: E402
from src.channels11 import compute_11_channels  # noqa: E402
from compare_multiscale import choose_pages  # noqa: E402


def evaluate_float(model, windows: np.ndarray):
    active = np.ones(len(windows), dtype=bool)
    scores = np.full(len(windows), -np.inf)
    trace = []
    for stage in model.stages:
        indices = np.flatnonzero(active)
        values = np.zeros(len(indices), dtype=np.float64)
        batch = windows[indices]
        for tree in stage.trees:
            def difference(node):
                feature = node.feature
                x1, y1 = feature.p1
                x2, y2 = feature.p2
                return batch[:, feature.channel, y1, x1] - batch[:, feature.channel, y2, x2]
            root = difference(tree.root) <= tree.root.threshold
            left = difference(tree.left) <= tree.left.threshold
            right = difference(tree.right) <= tree.right.threshold
            leaf = np.where(root, np.where(left, 0, 1), np.where(right, 2, 3))
            values += stage.learning_rate * np.asarray(tree.leaf_scores)[leaf]
        scores[indices] = values
        active[indices] = values >= stage.threshold
        trace.append({"entering": len(indices), "passing": int(active.sum())})
    return active, scores, trace


def float_channels(gray: np.ndarray) -> np.ndarray:
    """Use float32 storage/operations and the same explicit integer rounding."""
    g = gray.astype(np.float32)
    c = np.zeros((11, *g.shape), dtype=np.float32)
    c[0] = g
    h, w = g.shape
    if h > 1 and w > 1:
        a = np.floor((g[:-1, :-1] + g[1:, :-1] + 1) / 2)
        b = np.floor((g[:-1, 1:] + g[1:, 1:] + 1) / 2)
        c[1, :-1, :-1] = np.floor((a + b + 1) / 2)
    sh, sw = h - 3, w - 3
    if sh > 0 and sw > 0:
        a = c[1, :sh, :sw]
        b = c[1, :sh, 2:2+sw]
        d = c[1, 2:2+sh, :sw]
        e = c[1, 2:2+sh, 2:2+sw]
        c[2, :sh, :sw] = np.floor((np.floor((a+d)/2) + np.floor((e+b+1)/2) + 1)/2)
        for index, difference in enumerate((b-a, d-a, e-a, d-b), start=3):
            c[index, :sh, :sw] = np.floor((difference+255)/2)
    lh, lw = h - 7, w - 7
    if lh > 0 and lw > 0:
        a = c[2, :lh, :lw]
        b = c[2, :lh, 4:4+lw]
        d = c[2, 4:4+lh, :lw]
        e = c[2, 4:4+lh, 4:4+lw]
        for index, difference in enumerate((b-a, d-a, e-a, d-b), start=7):
            c[index, :lh, :lw] = np.floor((difference+255)/2)
    return c


def main() -> None:
    model_path = ROOT / "models/step4_cascade.json"
    model_bytes = model_path.read_bytes()
    model = cascade_from_dict(json.loads(model_bytes))
    manifest = ROOT / "datasets/manifests/normalized/manga109_faces.jsonl"
    rng = np.random.default_rng(20260925)
    images = []
    for page in choose_pages(manifest, 2):
        gray = cv2.imread(str(ROOT / "datasets" / page["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(page["image_path"])
        for _ in range(50):
            x = int(rng.integers(0, gray.shape[1]-23))
            y = int(rng.integers(0, gray.shape[0]-23))
            images.append(gray[y:y+24, x:x+24])
    start = perf_counter()
    integer = np.stack([compute_11_channels(image) for image in images])
    integer_seconds = perf_counter() - start
    start = perf_counter()
    floating = np.stack([float_channels(image) for image in images])
    float_seconds = perf_counter() - start
    a, scores_a, trace_a = model.evaluate(integer)
    b, scores_b, trace_b = evaluate_float(model, floating)
    report = {"model_sha256": hashlib.sha256(model_bytes).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "selection": "two fixed validation pages, 50 random 24x24 windows each",
              "seed": 20260925, "windows": len(images),
              "float_definition": "float32 storage and arithmetic with identical floor/rounding and boundary rules",
              "channel_max_abs_difference": float(np.max(np.abs(integer.astype(np.float32)-floating))),
              "pass_disagreements": int(np.count_nonzero(a != b)),
              "score_max_abs_difference": float(np.max(np.abs(scores_a-scores_b))),
              "integer_stage_counts": trace_a, "float_stage_counts": trace_b,
              "integer_channel_seconds": integer_seconds, "float_channel_seconds": float_seconds}
    target = ROOT / "results/step7_integer_float.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
