"""Render median, best, and worst validation teacher comparisons locally.

Licensed source crops remain in ignored tmp; the index carries source IDs,
model and annotation hashes, NME, and the exact sample selection policy.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landmark_regression import LandmarkRegressor
from src.landmark_hog import HogLandmarkRegressor
from scripts.iterate_landmark_teacher import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT/'results/landmark_teacher_iterations.json')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/landmark_iteration/visual_cases')
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding='utf-8'))
    model_path = ROOT/metrics['selected_by_validation']
    trials = metrics.get('trials') or [t for r in metrics['rounds'] for t in r['trials']]
    trial = next(t for t in trials if t['model'] == metrics['selected_by_validation'])
    if digest(model_path) != trial['sha256']:
        raise ValueError('Model changed since evaluation')
    source = ROOT/'datasets/annotations/auto/hrnetv2_full_20260927/annotations.jsonl'
    if digest(source) != metrics['annotation_sha256']:
        raise ValueError('Teacher annotations changed since evaluation')
    records = {r['image_id']: r for r in map(json.loads, source.read_text(encoding='utf-8').splitlines())}
    ordered = sorted(trial['validation']['per_image'], key=lambda r: (r['teacher_nme'], r['image_id']))
    middle = len(ordered)//2
    selections = [('best', ordered[:4]), ('median', ordered[middle-2:middle+2]), ('worst', ordered[-4:])]
    model = (HogLandmarkRegressor if model_path.suffix == '.joblib' else LandmarkRegressor).load(model_path)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    index = {'model': trial['model'], 'model_sha256': trial['sha256'],
             'annotation_sha256': metrics['annotation_sha256'], 'split': 'validation',
             'selection': '4 smallest, 4 around median, 4 largest teacher NME; no exclusions',
             'symbols': 'green circles: HRNet truth; magenta crosses: student; white segments: error',
             'review_status': 'generated_not_yet_visually_reviewed', 'cases': []}
    for category, cases in selections:
        tiles = []
        for number, case in enumerate(cases):
            row = records[case['image_id']]
            ann = row['annotations'][0]
            image = cv2.imread(str(ROOT/'datasets'/row['image_path']))
            if image is None:
                raise ValueError('Missing case image')
            prediction = model.predict_image(image, np.array(ann['bbox']))
            truth = np.array([[p['x'], p['y']] for p in ann['landmarks']])
            height, width = image.shape[:2]
            scale = 384/max(height, width)
            canvas = cv2.resize(image, (round(width*scale), round(height*scale)))
            for target, pred in zip(truth*scale, prediction*scale):
                t, p = tuple(np.rint(target).astype(int)), tuple(np.rint(pred).astype(int))
                cv2.line(canvas, t, p, (255, 255, 255), 1)
                cv2.circle(canvas, t, 3, (0, 255, 0), 1)
                cv2.drawMarker(canvas, p, (255, 0, 255), cv2.MARKER_CROSS, 7, 1)
            tile = np.full((430, 384, 3), 35, np.uint8)
            tile[:canvas.shape[0], :canvas.shape[1]] = canvas
            cv2.putText(tile, f'{category} {number+1}: NME {case["teacher_nme"]:.4f}',
                        (8, 410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            tiles.append(tile)
            index['cases'].append({**case, 'category': category, 'tile': number+1,
                                    'source_image': row['image_path'], 'sheet': f'{category}.jpg'})
        if not cv2.imwrite(str(output/f'{category}.jpg'), np.concatenate(tiles, axis=1)):
            raise ValueError('Could not write image sheet')
    (output/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
