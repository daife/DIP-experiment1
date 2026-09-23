"""Generate the six required original-plus-11-channel observation figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.channels11 import compute_11_channels  # noqa: E402


CHANNEL_TITLES = (
    "C0 grayscale",
    "C1 2x2 mean",
    "C2 gap-2 smooth",
    "C3 short horizontal",
    "C4 short vertical",
    "C5 short main diagonal",
    "C6 short anti-diagonal",
    "C7 long horizontal",
    "C8 long vertical",
    "C9 long main diagonal",
    "C10 long anti-diagonal",
)


@dataclass(frozen=True)
class Sample:
    slug: str
    category: str
    path: Path
    source: str
    crop_xyxy: tuple[int, int, int, int] | None = None
    download_url: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "step2_channels",
        help="Directory for source portraits, figures, and sample metadata.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Resize large inputs so their longest side is at most this value.",
    )
    return parser.parse_args()


def samples(output_dir: Path) -> list[Sample]:
    source_dir = output_dir / "source_images"
    return [
        Sample(
            "real_obama",
            "real frontal face 1",
            source_dir / "obama.jpg",
            "face_recognition example image (Obama)",
            (250, 20, 680, 450),
            "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        ),
        Sample(
            "real_biden",
            "real frontal face 2",
            source_dir / "biden.jpg",
            "face_recognition example image (Biden)",
            (250, 40, 750, 540),
            "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        ),
        Sample(
            "anime_front_1",
            "near-frontal anime face 1",
            ROOT / "datasets" / "raw" / "anime256" / "1000-2.jpg",
            "anime256:1000-2.jpg",
        ),
        Sample(
            "anime_front_2",
            "near-frontal anime face 2",
            ROOT / "datasets" / "raw" / "anime256" / "1009-8.jpg",
            "anime256:1009-8.jpg",
        ),
        Sample(
            "anime_hair_occlusion",
            "anime face with hair occlusion",
            ROOT / "datasets" / "raw" / "anime256" / "1007-4.jpg",
            "anime256:1007-4.jpg",
        ),
        Sample(
            "manga_complex_background",
            "manga complex background",
            ROOT
            / "datasets"
            / "raw"
            / "manga109"
            / "images"
            / "AisazuNihaIrarenai"
            / "010.jpg",
            "Manga109/AisazuNihaIrarenai/010.jpg",
        ),
    ]


def download_if_needed(sample: Sample) -> None:
    if sample.path.exists() or sample.download_url is None:
        return
    sample.path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        sample.download_url,
        headers={"User-Agent": "experiment1-channel-visualizer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        sample.path.write_bytes(response.read())


def read_bgr(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def prepare_image(sample: Sample, max_side: int) -> np.ndarray:
    image = read_bgr(sample.path)
    if sample.crop_xyxy is not None:
        x1, y1, x2, y2 = sample.crop_xyxy
        image = image[y1:y2, x1:x2]
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def save_figure(sample: Sample, bgr: np.ndarray, output_path: Path) -> None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    channels = compute_11_channels(gray)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    flat_axes = axes.ravel()
    flat_axes[0].imshow(rgb)
    flat_axes[0].set_title(f"Original: {sample.category}", fontsize=10)
    for axis, channel, title in zip(
        flat_axes[1:], channels, CHANNEL_TITLES, strict=True
    ):
        axis.imshow(channel, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=10)
    for axis in flat_axes:
        axis.axis("off")
    figure.suptitle(f"11-channel features — {sample.slug}", fontsize=15)
    figure.savefig(output_path, dpi=130, facecolor="white")
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_side < 24:
        raise ValueError("--max-side must be at least 24")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, object]] = []
    for sample in samples(output_dir):
        download_if_needed(sample)
        if not sample.path.exists():
            raise FileNotFoundError(
                f"Required local dataset image is missing: {sample.path}"
            )
        bgr = prepare_image(sample, args.max_side)
        output_path = output_dir / f"{sample.slug}_channels.png"
        save_figure(sample, bgr, output_path)
        metadata.append(
            {
                "slug": sample.slug,
                "category": sample.category,
                "source": sample.source,
                "source_path": str(sample.path.relative_to(ROOT))
                if sample.path.is_relative_to(ROOT)
                else str(sample.path),
                "source_sha256": sha256(sample.path),
                "crop_xyxy": sample.crop_xyxy,
                "processed_shape": list(bgr.shape),
                "figure": str(output_path.relative_to(ROOT)),
            }
        )
        print(f"wrote {output_path}")

    metadata_path = output_dir / "samples.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
