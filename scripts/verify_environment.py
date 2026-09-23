"""Verify that the experiment environment is ready for subsequent steps.

This check intentionally avoids downloading model weights or opening GUI windows.
It validates imports and a small representative operation for each core dependency.
"""

from __future__ import annotations

import importlib
import io
import json
import platform
import sys
from typing import Any, Callable


MINIMUM_PYTHON = (3, 12)


def version_of(module: Any) -> str:
    """Return a human-readable package version when one is exposed."""

    return str(getattr(module, "__version__", "unknown"))


def check_numpy() -> tuple[str, str]:
    np = importlib.import_module("numpy")
    values = np.array([0, 255], dtype=np.uint8).astype(np.int16)
    if int(values[1] - values[0]) != 255:
        raise RuntimeError("int16 pixel subtraction produced an unexpected result")
    return version_of(np), "uint8-to-int16 pixel arithmetic"


def check_opencv() -> tuple[str, str]:
    cv2 = importlib.import_module("cv2")
    np = importlib.import_module("numpy")
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape != (4, 4) or gray.dtype != np.uint8:
        raise RuntimeError("OpenCV grayscale conversion returned an unexpected result")
    return version_of(cv2), "BGR-to-grayscale conversion"


def check_sklearn() -> tuple[str, str]:
    sklearn = importlib.import_module("sklearn")
    tree_module = importlib.import_module("sklearn.tree")
    classifier = tree_module.DecisionTreeClassifier(max_depth=2, random_state=0)
    classifier.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    if classifier.get_depth() > 2:
        raise RuntimeError("Depth-2 decision-tree smoke test failed")
    return version_of(sklearn), "Depth-2 decision-tree fit"


def check_matplotlib() -> tuple[str, str]:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")
    figure = pyplot.figure(figsize=(1, 1))
    pyplot.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    pyplot.close(figure)
    if buffer.tell() == 0:
        raise RuntimeError("Matplotlib failed to render a PNG in headless mode")
    return version_of(matplotlib), "headless PNG rendering"


def check_pillow() -> tuple[str, str]:
    pil = importlib.import_module("PIL")
    image_module = importlib.import_module("PIL.Image")
    image = image_module.new("L", (4, 4), color=127)
    if image.getpixel((0, 0)) != 127:
        raise RuntimeError("Pillow image smoke test failed")
    return version_of(pil), "grayscale image creation"


def check_tqdm() -> tuple[str, str]:
    tqdm_module = importlib.import_module("tqdm")
    list(tqdm_module.tqdm(range(1), disable=True))
    return version_of(tqdm_module), "disabled progress iteration"


def check_anime_face_detector() -> tuple[str, str]:
    detector_module = importlib.import_module("anime_face_detector")
    torch = importlib.import_module("torch")
    required_symbols = ("LandmarkDetector", "get_checkpoint_path")
    missing = [name for name in required_symbols if not hasattr(detector_module, name)]
    if missing:
        raise RuntimeError(f"anime-face-detector is missing symbols: {missing}")
    if torch.tensor([1, 2]).sum().item() != 3:
        raise RuntimeError("PyTorch tensor smoke test failed")
    return version_of(torch), "anime detector API and CPU tensor operation"


CHECKS: dict[str, Callable[[], tuple[str, str]]] = {
    "numpy": check_numpy,
    "opencv": check_opencv,
    "scikit-learn": check_sklearn,
    "matplotlib": check_matplotlib,
    "pillow": check_pillow,
    "tqdm": check_tqdm,
    "anime-face-detector/torch": check_anime_face_detector,
}


def main() -> int:
    report: dict[str, Any] = {
        "status": "ok",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "checks": {},
    }
    failures: list[str] = []

    if sys.version_info < MINIMUM_PYTHON:
        failures.append(
            f"Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer is required"
        )

    for name, check in CHECKS.items():
        try:
            version, operation = check()
            report["checks"][name] = {
                "status": "ok",
                "version": version,
                "operation": operation,
            }
        except Exception as exc:  # report all missing/broken dependencies at once
            failures.append(f"{name}: {exc}")
            report["checks"][name] = {"status": "failed", "error": repr(exc)}

    if failures:
        report["status"] = "failed"
        report["failures"] = failures

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
