"""Compare box regression families with fixed training data and verifier threshold."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import CandidateVerifier, features
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig, detect_multiscale, nms
from src.detection_metrics import match_detections
from compare_multiscale import choose_pages
from evaluate_box_refiner import transform


def main():
    cache = ROOT / 'datasets/derived/step8_box_refiner_training/samples.npz'
    metadata = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    if any(p['split'] != 'train' for p in metadata['provenance']):
        raise ValueError('regressor training must use train only')
    if hashlib.sha256(cache.read_bytes()).hexdigest() != metadata['samples_sha256']:
        raise ValueError('training cache hash mismatch')
    with np.load(cache) as data:
        x, y = data['x'], data['y']
    target_boxes = np.asarray([p['target'] for p in metadata['provenance']])
    proposal_boxes = np.asarray([p['proposal'] for p in metadata['provenance']])
    size_ratios = (target_boxes[:, 2:] - target_boxes[:, :2]) / (proposal_boxes[:, 2:] - proposal_boxes[:, :2])
    coverage = {'annotated_faces': sum(p['gt'] for p in metadata['pages']),
                'represented_faces': len({(p['image_id'], p['gt_index']) for p in metadata['provenance']}),
                'size_ratio_quantiles': np.quantile(size_ratios, [0, .1, .5, .9, 1], axis=0).tolist(),
                'quantile_levels': [0, .1, .5, .9, 1]}
    ridge_path = ROOT / 'models/box_refiner_ridge_v1.joblib'
    regressors = {'ridge': joblib.load(ridge_path)}
    for gamma in (0.1, 0.5):
        name = f'kernel_ridge_gamma{gamma:g}'
        regressors[name] = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma).fit(x, y)
        joblib.dump(regressors[name], cache.parent / (name + '.joblib'), compress=3)
    cascade_path = ROOT / 'models/step4_cascade.json'
    manifest = ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl'
    verifier_path = ROOT / 'models/candidate_hog_rbf_step3_v1.joblib'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata['manifest_sha256']:
        raise ValueError('manifest changed since training')
    if hashlib.sha256(cascade_path.read_bytes()).hexdigest() != metadata['cascade_sha256']:
        raise ValueError('cascade changed since training')
    cascade = cascade_from_dict(json.loads(cascade_path.read_text()))
    verifier = CandidateVerifier(verifier_path)
    totals = {name: dict(tp=0, fp=0, fn=0) for name in regressors}
    pages = []
    for page in choose_pages(manifest, 8):
        start = perf_counter()
        gray = cv2.imread(str(ROOT / 'datasets' / page['image_path']), 0)
        boxes, _, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
        x_page = features(gray, boxes)
        scores = verifier.model.decision_function(x_page)
        keep = (scores >= 1.5) & ((boxes[:, 2] - boxes[:, 0]) >= 36)
        boxes, scores, x_page = boxes[keep], scores[keep], x_page[keep]
        gt = np.asarray([a['bbox'] for a in page['annotations']])
        item = {'image_id': page['image_id'], 'results': {}}
        for name, model in regressors.items():
            if len(boxes):
                adjusted, valid = transform(boxes, model.predict(x_page), gray.shape[1], gray.shape[0])
                adjusted, selected_scores = adjusted[valid], scores[valid]
                indices = nms(adjusted, selected_scores, 0.3)
                result = match_detections(adjusted[indices], selected_scores[indices], gt)
            else:
                result = dict(tp=0, fp=0, fn=len(gt))
            item['results'][name] = {k: result[k] for k in ('tp', 'fp', 'fn')}
            for k in ('tp', 'fp', 'fn'):
                totals[name][k] += result[k]
        item['seconds_all_variants'] = perf_counter() - start
        pages.append(item)
        print(item, flush=True)
    for t in totals.values():
        t['precision'] = t['tp'] / max(1, t['tp'] + t['fp'])
        t['recall'] = t['tp'] / max(1, t['tp'] + t['fn'])
        t['f1'] = 2*t['tp'] / max(1, 2*t['tp'] + t['fp'] + t['fn'])
    report = {'split': 'validation', 'threshold': 1.5, 'final_nms_iou': 0.3,
              'training_samples': len(y), 'training_coverage': coverage, 'kernel_alpha': 1.0, 'kernel_gamma': [0.1, 0.5],
              'selection': 'fixed step5-comparison SHA-256 order; eight distinct validation groups',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (cache, cache.with_suffix('.json'), ridge_path, manifest, cascade_path, verifier_path)},
              'pages': pages, 'summary': totals}
    (ROOT / 'results/step8_box_regression_family_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(totals)


if __name__ == '__main__':
    main()
