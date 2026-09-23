# Derived data workspace

Recommended subdirectories:

- `detection_crops/`: positive 24 x 24 face crops plus crop metadata linking back to the source image and box.
- `negatives/`: negative 24 x 24 windows sampled from real LAP/Manga109 scenes.
- `hard_negatives/`: false positives mined by a trained cascade.

All generated samples must inherit the source image split from `manifests/images.csv`. Never resplit crops independently.
