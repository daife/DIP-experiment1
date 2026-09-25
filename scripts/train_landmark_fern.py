"""Train and evaluate the optional Fern-style landmark comparison."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_landmark_regressor import ANNOTATIONS, MEAN, evaluate, load_data  # noqa: E402
from src.landmark_fern import FernLandmarkRegressor, train_fern  # noqa: E402
from src.landmark_regression import make_offsets  # noqa: E402


def main() -> None:
    mean_data = json.loads(MEAN.read_text(encoding="utf-8"))
    annotation_hash = hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()
    if mean_data["source_sha256"] != annotation_hash:
        raise ValueError("Mean shape source does not match annotations")
    data = load_data()
    offsets = make_offsets(20260925)
    mask = data["train"]["visible"]
    model = train_fern(data["train"]["images"], data["train"]["targets"], mask,
                       np.asarray(mean_data["points_xy"], dtype=np.float64), offsets)
    path = ROOT / "models/landmark_fern4_visible_only.npz"
    path.parent.mkdir(exist_ok=True)
    model.save(path)
    loaded = FernLandmarkRegressor.load(path)
    results = {
        "method": "four-stage local binary fern (two four-bit codes per point, one-hot leaves, masked per-point ridge)",
        "annotation_sha256": annotation_hash,
        "mean_shape_sha256": hashlib.sha256(MEAN.read_bytes()).hexdigest(),
        "seed": 20260925, "stages": 4, "differences_per_point": 8,
        "fern_bits": 4, "ferns_per_point": 2, "ridge_alpha": 10.0,
        "offset_radius_bbox_fraction": 0.075, "image_normalization_size": 256,
        "train_target_points": int(mask.sum()), "train_target_images": int(mask.any(axis=1).sum()),
        "target_status": "coarse-human-accepted, unchanged HRNetV2 coordinates",
        "model": str(path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation": evaluate(loaded, data["validation"]),
        "test": evaluate(loaded, data["test"]),
    }
    output = ROOT / "results/landmark_fern_coarse_metrics.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("validation", results["validation"]["visible_nme"], "test", results["test"]["visible_nme"])


if __name__ == "__main__":
    main()
