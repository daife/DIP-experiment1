#!/usr/bin/env python3
"""Calculate 11 uint8 channels for every generated 24x24 detection crop."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_detection_dataset import ROOT, SIZE
sys.path.insert(0, str(ROOT))
from src.channels11 import compute_11_channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/derived/step3_detection_v1")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    channels = np.lib.format.open_memmap(args.dataset / "channels11.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(rows), 11, SIZE, SIZE))
    with (args.dataset / "channels11_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["array_index", "sample_id"])
        for index, row in enumerate(rows):
            with Image.open(args.dataset / row["file"]) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
            channels[index] = np.stack(compute_11_channels(gray))
            writer.writerow([index, row["sample_id"]])
    channels.flush()
    print(f"Saved {channels.shape} uint8 channels to {args.dataset / 'channels11.npy'}")


if __name__ == "__main__":
    main()
