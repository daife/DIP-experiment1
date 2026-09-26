"""Train localization quality on train face proposals and evaluate NMS ranking."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import CandidateVerifier, features
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig, detect_multiscale, nms
from src.detection_metrics import match_detections
from compare_multiscale import choose_pages
from evaluate_box_refiner import transform


def paired_iou(boxes, targets):
    wh = np.maximum(0, np.minimum(boxes[:, 2:], targets[:, 2:]) - np.maximum(boxes[:, :2], targets[:, :2]))
    intersection = np.prod(wh, axis=1)
    return intersection / (np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) + np.prod(targets[:, 2:] - targets[:, :2], axis=1) - intersection)


def main():
    cache = ROOT / 'datasets/derived/step8_box_refiner_training/samples.npz'
    metadata = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    manifest = ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl'
    cascade_path = ROOT / 'models/step4_cascade.json'
    refiner_path = ROOT / 'models/box_refiner_ridge_v1.joblib'
    verifier_path = ROOT / 'models/candidate_hog_rbf_step3_v1.joblib'
    for path, expected in ((cache, metadata['samples_sha256']), (manifest, metadata['manifest_sha256']), (cascade_path, metadata['cascade_sha256']), (refiner_path, metadata['model_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'input hash mismatch: {path}')
    if any(p['split'] != 'train' for p in metadata['provenance']):
        raise ValueError('ranker training requires train only')
    with np.load(cache) as data:
        x = data['x']
    refiner = joblib.load(refiner_path)
    boxes = np.asarray([p['proposal'] for p in metadata['provenance']])
    targets = np.asarray([p['target'] for p in metadata['provenance']])
    # Clip each proposal to its actual source image bounds, as in inference.
    rows = {r['image_id']: r for r in map(json.loads, manifest.read_text(encoding='utf-8').splitlines())}
    adjusted = np.empty_like(boxes)
    for image_id in dict.fromkeys(p['image_id'] for p in metadata['provenance']):
        indices = [i for i,p in enumerate(metadata['provenance']) if p['image_id'] == image_id]
        gray = cv2.imread(str(ROOT / 'datasets' / rows[image_id]['image_path']), 0)
        adjusted[indices], valid = transform(boxes[indices], refiner.predict(x[indices]), gray.shape[1], gray.shape[0])
        if not valid.all():
            raise ValueError('invalid refined training proposal')
    quality = paired_iou(adjusted, targets)
    ranker = Ridge(alpha=10).fit(x, quality)
    output_dir = ROOT / 'datasets/derived/step8_localization_ranker'
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker_path = output_dir / 'ridge_quality.joblib'
    joblib.dump(ranker, ranker_path, compress=3)
    cascade = cascade_from_dict(json.loads(cascade_path.read_text()))
    verifier = CandidateVerifier(verifier_path)
    totals = {name: dict(tp=0, fp=0, fn=0) for name in ('rbf', 'quality', 'rbf_times_quality')}
    pages = []
    for page in choose_pages(manifest, 8):
        start = perf_counter()
        gray = cv2.imread(str(ROOT / 'datasets' / page['image_path']), 0)
        boxes, _, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, .5))
        x_page = features(gray, boxes)
        scores = verifier.model.decision_function(x_page)
        keep = (scores >= 1.5) & ((boxes[:, 2] - boxes[:, 0]) >= 36)
        boxes, scores, x_page = boxes[keep], scores[keep], x_page[keep]
        gt = np.asarray([a['bbox'] for a in page['annotations']])
        adjusted, valid = transform(boxes, refiner.predict(x_page), gray.shape[1], gray.shape[0])
        adjusted, scores, x_page = adjusted[valid], scores[valid], x_page[valid]
        predicted = np.clip(ranker.predict(x_page), 0, 1)
        np.savez_compressed(output_dir / (page['image_id'].replace(':', '_') + '.npz'), boxes=adjusted, scores=scores, features=x_page, gt=gt)
        item = {'image_id': page['image_id'], 'results': {}}
        for name, ranking in (('rbf', scores), ('quality', predicted), ('rbf_times_quality', scores*predicted)):
            indices = nms(adjusted, ranking, .3)
            result = match_detections(adjusted[indices], ranking[indices], gt)
            item['results'][name] = {k: result[k] for k in ('tp','fp','fn')}
            for k in ('tp','fp','fn'):
                totals[name][k] += result[k]
        item['seconds_all_variants'] = perf_counter() - start
        pages.append(item)
        print(item, flush=True)
    for t in totals.values():
        t['precision'] = t['tp'] / max(1,t['tp']+t['fp'])
        t['recall'] = t['tp'] / max(1,t['tp']+t['fn'])
        t['f1'] = 2*t['tp'] / max(1,2*t['tp']+t['fp']+t['fn'])
    report = {'split':'validation', 'train_samples':len(x), 'alpha':10, 'first_nms':.5, 'final_nms':.3, 'threshold':1.5,
              'train_quality_quantiles': np.quantile(quality,[0,.1,.5,.9,1]).tolist(),
              'training_limitation':'Quality targets use in-sample refined train face proposals; no background quality labels; not calibrated face probabilities',
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cache,cache.with_suffix('.json'),manifest,cascade_path,refiner_path,verifier_path,ranker_path)},
              'pages':pages,'summary':totals}
    (ROOT / 'results/step8_localization_ranker_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(totals)


if __name__ == '__main__':
    main()
