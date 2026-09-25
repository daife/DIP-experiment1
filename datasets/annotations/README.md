# Annotation workspace

- `auto/`: write each preannotation run to its own versioned subdirectory, including model name, model checksum, parameters, and run date.
- `corrected/`: human-reviewed 28-point labels. Never overwrite an automatic run in place.
- `review_sets/landmark320/`: fixed 320-image preannotation cohort, image hashes, and the versioned review images used by the local tool.

Each corrected record must keep `image_id`, `source_dataset`, `source_group`, and `split` from `manifests/images.csv`; contain exactly 28 landmarks in the agreed fixed order; and store `visibility` as 0 or 1. Preserve model confidence separately from human visibility.

The fixed zero-based order and flip pairs are in `landmark28_schema.json`; the numbered schematic is `../../results/landmark28_index.png`. The array order matches `anime-face-detector/hrnetv2` directly. Image-left and image-right refer to the viewer's image coordinates, not anatomical left/right.

Automatic predictions retain `visibility: null` until a person checks each point. A corrected annotation records 0 for an occluded, cropped, or otherwise unlocatable point and 1 for a visible, human-checked point. Retain all 28 array positions and the separate model confidence. Mask visibility-0 points during training and NME. Test-split annotations require human review before evaluation.

The local review UI requires an explicit save for each inspected image. At save time, points still carrying the automatic `null` visibility become 1; points marked 0 remain masked. The record stores a save timestamp, method, edited point indices, and occluded point indices without asking the reviewer to type an identifier. An entirely unlabelable image requires a note explaining why.

An anatomically plausible prediction behind hair is still visibility 0: plausibility is not visual evidence. Do not move an occluded point to a guessed coordinate; its coordinate remains a masked placeholder. Move a point and mark visibility 1 only when its target location can actually be identified in the image.
