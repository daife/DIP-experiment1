# Dataset workspace

This directory separates immutable downloads from extracted source data and generated metadata.

## Layout

- `downloads/`: original archives; do not edit their contents.
- `raw/`: extracted source datasets, kept in their native layouts.
- `manifests/`: generated inventories, validation reports, and normalized metadata.
- `staging/`: temporary nested archives used during extraction.

## Intended roles

- `LAP`: full poster images plus face/head bounding boxes. Use for face detection positives, scene negatives, and hard-negative mining. Prefer manually refined `labels_faces.txt` where available; retain the automatic baseline separately with its confidence score.
- `Manga109`: full manga pages plus XML annotations. Use face boxes as detection positives and pages as the main source of realistic negatives. Split by manga title, never by crop or page.
- `anime256`: already-cropped 256 x 256 face images. Use as a positive-image and landmark-preannotation pool, not as background negatives. It has no native bounding-box or landmark labels.

Generated crops, preannotations, corrections, and train/validation/test splits should live outside `raw/`, so the original datasets remain reproducible.

## Generated manifests

- `manifests/images.csv`: one row per usable source image, including dimensions, source group, deterministic split, and label availability.
- `manifests/groups.csv`: the fixed group-level split assignment.
- `manifests/normalized/lap_faces_manual.jsonl`: the first 5,000 LAP poster records with manually refined boxes and 15-way face classes.
- `manifests/normalized/lap_faces_baseline.jsonl`: automatic LAP face boxes for all posters; the fifth source value is preserved as confidence, not a class ID.
- `manifests/normalized/manga109_faces.jsonl`: face boxes from the current Manga109-v2026 XML annotations.
- `manifests/annotation_schema.json`: target schema for subsequent 28-point preannotation and correction.
- `manifests/validation_errors.txt`: source-label anomalies found during validation. These are reported rather than silently clipped or rewritten.
- `manifests/SHA256SUMS.txt`: archive integrity hashes.

Regenerate manifests with:

```powershell
python datasets/tools/build_manifests.py
```

The fixed split seed is `experiment1-anime-face-v1`. LAP is grouped by MAL title ID, Manga109 by book title, and anime256 by the recoverable original-image token before the final crop index. This preserves the required 75/10/15 split policy without allowing related samples to cross partitions.

## Preannotation staging

Create new outputs under the following directories when the landmark pipeline is added:

```text
annotations/
  auto/       # model output, immutable after a run
  corrected/  # human-reviewed 28-point labels
derived/
  detection_crops/
  negatives/
  hard_negatives/
```

Do not treat anime256 as if it has ground-truth landmarks: it is a crop pool only. For LAP, prefer manual boxes where present and fall back to baseline boxes with an explicit confidence threshold. For Manga109, pages with zero face annotations are especially useful for negative-window sampling, but sampling must stay within the page's assigned split.
