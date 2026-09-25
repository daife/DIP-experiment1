"""Run the exported detector and save a visualization plus JSON."""

import argparse
import json
from pathlib import Path

import cv2

from src.detector import AnimeFaceDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    detections = AnimeFaceDetector(args.model_dir).detect(image)
    canvas = image.copy()
    for result in detections:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        cv2.putText(canvas, f'{result["score"]:.3f}', (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        for point in result["landmarks"]:
            cv2.circle(canvas, tuple(round(value) for value in point), 2, (0, 0, 255), -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"cannot write image: {args.output}")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(detections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"detections": len(detections), "image": str(args.output), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
