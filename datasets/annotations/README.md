# Annotation workspace

- `auto/`: write each preannotation run to its own versioned subdirectory, including model name, model checksum, parameters, and run date.
- `corrected/`: human-reviewed 28-point labels. Never overwrite an automatic run in place.

Each corrected record must keep `image_id`, `source_dataset`, `source_group`, and `split` from `manifests/images.csv`; contain exactly 28 landmarks in the agreed fixed order; and store `visibility` as 0 or 1. Preserve model confidence separately from human visibility.

