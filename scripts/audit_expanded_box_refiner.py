"""Independently verify saved regression targets against original train GT."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    folder = ROOT / 'datasets/derived/cascade_auto_box_refiner_expanded_v1'
    pool = ROOT / 'datasets/derived/cascade_auto_scan_positives_merged_v2'
    manifest = ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl'
    meta = json.loads((folder / 'provenance.json').read_text())
    for name, key in [('features_targets.npz', 'features_targets_sha256'),
                      ('samples.jsonl', 'samples_sha256'), ('model.joblib', 'model_sha256')]:
        assert sha(folder / name) == meta[key]
    assert sha(manifest) == meta['manifest_sha256']
    pages = {r['image_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    source = list(map(json.loads, (pool / 'candidates.jsonl').read_text().splitlines()))
    saved = list(map(json.loads, (folder / 'samples.jsonl').read_text().splitlines()))
    data = np.load(folder / 'features_targets.npz')
    assert np.array_equal(data['x'], np.load(pool / 'hog_features.npy'))
    assert len(source) == len(saved) == len(data['y']) == 12817
    for index, (row, sample, target) in enumerate(zip(source, saved, data['y'])):
        page = pages[row['page_id']]
        assert page['split'] == sample['split'] == row['split'] == 'train'
        assert row['label'] == 1 and sample['source_index'] == index
        assert sample['source_id'] == row['id'] and sample['page_id'] == row['page_id']
        box = np.array(row['bbox_xyxy'], dtype=float)
        gt = np.array([a['bbox'] for a in page['annotations']], dtype=float)
        intersection = np.prod(np.maximum(0, np.minimum(box[2:], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2])), axis=1)
        areas = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
        width, height = box[2:] - box[:2]
        overlap = intersection / (areas + width * height - intersection)
        chosen = int(overlap.argmax())
        assert chosen == sample['annotation_index'] and overlap[chosen] >= .5
        assert np.array_equal(box, sample['proposal'])
        assert np.array_equal(gt[chosen], sample['target'])
        expected = np.r_[((gt[chosen, :2] + gt[chosen, 2:]) / 2 - (box[:2] + box[2:]) / 2) / [width, height],
                         np.log((gt[chosen, 2:] - gt[chosen, :2]) / [width, height])]
        assert np.array_equal(expected, target)
    report = {'samples': len(saved), 'pages': len({r['page_id'] for r in saved}),
              'all_train_GT_targets_exact': True, 'all_source_HOG_exact': True,
              'saved_artifact_hashes_match': True, 'provenance_sha256': sha(folder / 'provenance.json'),
              'manifest_sha256': sha(manifest), 'script_sha256': sha(Path(__file__)),
              'limitation': 'Numerical provenance audit; no new visual review or detector acceptance.'}
    output = ROOT / 'results/cascade_auto_box_refiner_expanded_v1_input_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
