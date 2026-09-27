"""Nested training-data expansion using unfiltered HRNet coordinates as truth.

Run only after preannotation metadata exists. Visibility stays unknown; this
experiment reports all-point teacher NME, not human-visible-point NME.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landmark_regression import LandmarkRegressor, make_offsets, normalized_image, train


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, data):
    predictions = np.concatenate([model.predict_normalized(data['images'][i:i+128])
                                  for i in range(0, len(data['images']), 128)])
    targets = data['targets']
    denominator = np.linalg.norm(targets[:, 11:17].mean(1) - targets[:, 17:23].mean(1), axis=1)
    fallback = denominator <= 1e-6
    denominator[fallback] = np.sqrt(2)
    errors = np.linalg.norm(predictions-targets, axis=2) / denominator[:, None]
    per_image = errors.mean(1)
    return {'images': len(targets), 'teacher_all_point_nme': float(per_image.mean()),
            'median_nme': float(np.median(per_image)), 'p90_nme': float(np.quantile(per_image, .9)),
            'fraction_nme_le_0_1': float((per_image <= .1).mean()),
            'bbox_diagonal_fallback_images': int(fallback.sum()),
            'per_image': [{'image_id': image_id, 'teacher_nme': float(nme)}
                          for image_id, nme in zip(data['ids'], per_image)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--round-sizes', type=int, nargs='+', default=[1000, 4000, 20710])
    args = parser.parse_args()
    if any(n <= 0 for n in args.round_sizes) or args.round_sizes != sorted(set(args.round_sizes)):
        raise ValueError('Round sizes must be positive, strictly increasing')
    metadata = json.loads((args.run/'run_metadata.json').read_text(encoding='utf-8'))
    if metadata['result']['failures']:
        raise ValueError('Resolve and document teacher failures before training')
    source = args.run/'annotations.jsonl'
    sets = {s: {'images': [], 'targets': [], 'ids': [], 'groups': set()} for s in ('train', 'validation', 'test')}
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    manifest_path = ROOT/'datasets/manifests/images.csv'
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        expected = {r['image_id']: r for r in csv.DictReader(handle) if r['source_dataset'] == 'anime256'}
    ids = [r['image_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Teacher data must contain every anime256 manifest ID exactly once')
    if metadata['result']['successful_images'] != len(records):
        raise ValueError('Teacher metadata count does not match records')
    for row in records:
        original = expected[row['image_id']]
        if any(row[k] != original[k] for k in ('image_path', 'split', 'source_group', 'source_dataset')):
            raise ValueError(f"Manifest provenance mismatch: {row['image_id']}")
    cache = ROOT/'tmp/landmark_iteration'
    cache.mkdir(parents=True, exist_ok=True)
    for split, item in sets.items():
        count = sum(row['split'] == split for row in records)
        item['images'] = np.memmap(cache/f'{split}_images.float32', mode='w+', dtype=np.float32,
                                   shape=(count, 256, 256))
    for row in records:
        item = sets[row['split']]
        ann = row['annotations'][0]
        box = np.array(ann['bbox'])
        image = cv2.imread(str(ROOT/'datasets'/row['image_path']), cv2.IMREAD_GRAYSCALE)
        xy = np.array([[p['x'], p['y']] for p in ann['landmarks']])
        if image is None or xy.shape != (28, 2) or not np.isfinite(xy).all():
            raise ValueError(f"Invalid teacher record: {row['image_id']}")
        item['images'][len(item['ids'])] = normalized_image(image, box)
        item['targets'].append((xy-box[:2])/(box[2:]-box[:2]))
        item['ids'].append(row['image_id'])
        item['groups'].add(row['source_group'])
    for s, item in sets.items():
        for other in sets:
            if s != other and item['groups'] & sets[other]['groups']:
                raise ValueError('Source group leaks across splits')
        item['images'].flush()
        item['targets'] = np.asarray(item['targets'])
    result = {'annotation_sha256': digest(source), 'manifest_sha256': digest(manifest_path),
              'teacher_metadata': metadata, 'integrity': 'complete unique manifest coverage; fixed provenance; disjoint source groups',
              'seed': 20260927, 'truth_policy': 'user-authorized HRNet absolute truth; no confidence filtering',
              'metric': 'all 28 points / teacher eye-center distance; NOT human visibility NME',
              'split_counts': {s: len(d['ids']) for s, d in sets.items()}, 'rounds': []}
    output = ROOT/'results/landmark_teacher_iterations.json'
    for count in args.round_sizes:
        count = min(count, len(sets['train']['ids']))
        images = sets['train']['images'][:count]
        targets = sets['train']['targets'][:count]
        mean = targets.mean(0)
        trials = []
        for differences, alpha in ((8, 10.), (16, 10.), (16, 1.)):
            model = train(images, targets, np.ones((count, 28), bool), mean,
                          make_offsets(20260927, differences), stages=5, alpha=alpha)
            path = ROOT/f'models/landmark_teacher_n{count}_f{differences}_a{alpha:g}.npz'
            model.save(path)
            restored = LandmarkRegressor.load(path)
            validation = evaluate(restored, sets['validation'])
            trials.append({'model': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                           'differences_per_point': differences, 'alpha': alpha, 'stages': 5,
                           'validation': validation})
            print(count, differences, alpha, validation['teacher_all_point_nme'], flush=True)
        result['rounds'].append({'train_images': count, 'train_ids': sets['train']['ids'][:count], 'trials': trials})
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    best = min((trial for r in result['rounds'] for trial in r['trials']),
               key=lambda trial: trial['validation']['teacher_all_point_nme'])
    result['selected_by_validation'] = best['model']
    result['selected_test'] = evaluate(LandmarkRegressor.load(ROOT/best['model']), sets['test'])
    result['historical_model_validation'] = evaluate(
        LandmarkRegressor.load(ROOT/'models/landmark_ridge4_visible_only.npz'), sets['validation'])
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
