# 320-image landmark review layers

- `annotations.jsonl`: HRNetV2 automatic preannotation, all visibility values `null`.
- `assisted_review_v2.jsonl`: the automated visual review submitted for human inspection; its own metadata explicitly says it was not human reviewed. SHA-256: `56ce64034a043b318adb8b4c6be53eb50c798de1beeef2b8c975c6646147dc0b`.
- `assisted_review_v2_summary.json`: original summary of that assisted pass. Its absolute `input`, `export_output`, and `schema` paths describe the machine that produced it; `output_sha256` matches the assisted JSONL kept here.
- `../../corrected/landmark28_review320.jsonl`: same point coordinates and V/H/U classes after the user confirmed a simple full-cohort visual review. The human acceptance and its limit are stored separately in `../../corrected/landmark28_review320_acceptance.json`.

Use `scripts/promote_landmark_review.py` to reproduce the accepted JSONL and `scripts/verify_landmark_review.py` to validate its lineage and counts. No coordinate was fine-tuned during the human pass. The six all-U images are unusable for shape fitting and NME until individually revisited.
