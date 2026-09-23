"""The training loader must honor manual review and channel-cache indices."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_cascade import load_reviewed_data


class TrainingDataTests(unittest.TestCase):
    def test_filtered_indices_follow_reviewed_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "review_summary.json").write_text('{"fully_reviewed": true}', encoding="utf-8")
            rows = [
                {"sample_id": "a", "split": "train", "label": 1, "review_decision": "keep"},
                {"sample_id": "c", "split": "validation", "label": 0, "review_decision": "keep"},
            ]
            (dataset / "usable_samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with (dataset / "usable_channels11_index.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerows((("array_index", "sample_id"), ("2", "a"), ("0", "c")))
            cache = np.zeros((3, 11, 24, 24), dtype=np.uint8)
            cache[0, 0, 0, 0] = 10
            cache[2, 0, 0, 0] = 20
            np.save(dataset / "channels11.npy", cache)
            data = load_reviewed_data(dataset)
            self.assertEqual(int(data["train"][0][0, 0, 0, 0]), 20)
            self.assertEqual(int(data["validation"][0][0, 0, 0, 0]), 10)
            (dataset / "review_summary.json").write_text('{"fully_reviewed": false}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reviewed_data(dataset)


if __name__ == "__main__":
    unittest.main()
