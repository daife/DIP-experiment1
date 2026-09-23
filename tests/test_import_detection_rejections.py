import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_detection_rejections import import_rejections


class ImportRejectionsTests(unittest.TestCase):
    def test_reject_only_list_marks_remaining_base_crops_keep(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = root / "samples.jsonl"
            samples.write_text("\n".join(json.dumps(row) for row in [
                {"sample_id": "trp00000", "augmentation": None},
                {"sample_id": "trp00000f", "augmentation": "horizontal_flip"},
                {"sample_id": "trn00000", "augmentation": None},
            ]) + "\n", encoding="utf-8")
            source = root / "reject.csv"
            source.write_text("trn00000,reject,,contains face\n", encoding="utf-8")
            output = root / "review.csv"
            self.assertEqual(import_rejections(source, samples, output),
                             {"base_samples": 2, "keep": 1, "reject": 1})
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([(row["sample_id"], row["decision"]) for row in rows],
                             [("trp00000", "keep"), ("trn00000", "reject")])
            self.assertEqual(rows[1]["note"], "contains face")

    def test_unknown_id_fails_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = root / "samples.jsonl"
            samples.write_text('{"sample_id":"trp00000","augmentation":null}\n', encoding="utf-8")
            source = root / "reject.csv"
            source.write_text("unknown,reject,,\n", encoding="utf-8")
            output = root / "review.csv"
            with self.assertRaisesRegex(ValueError, "Unknown IDs"):
                import_rejections(source, samples, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
