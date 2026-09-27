"""Trace refined validation misses and test the existing minimum size gate."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_box_refiner import ious
from scripts.evaluate_box_refiner import transform
from src.multiscale import nms
from src.detection_metrics import match_detections

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = ROOT / 'results/cascade_auto_nms05_expanded_refiner_v1_combined20.json'
    report = json.loads(source.read_text())
    directory = ROOT / 'datasets/derived/cascade_auto_scan_positive_localization_v1'
    cache_dir = directory / 'nms05_cache'
    prediction_path = directory / 'nms05_expanded_refiner_v1/predictions.json'
    predictions = json.loads(prediction_path.read_text())
    assert [p['image_id'] for p in predictions] == [p['image_id'] for p in report['pages']]
    assert json.loads((cache_dir / 'source_state.json').read_text()) == {p.name: sha(p) for p in (ROOT / 'src').glob('*.py')}
    for key, value in report['sha256'].items():
        assert sha(ROOT / key) == value
    caches = {sha(p): p for p in cache_dir.glob('*.npz')}
    verifier = joblib.load(ROOT / 'datasets/derived/cascade_auto_proposal_verifier_v5/model.joblib')
    refiner = joblib.load(ROOT / 'datasets/derived/cascade_auto_box_refiner_expanded_v1/model.joblib')
    details = []
    totals = {str(side): dict(tp=0, fp=0, fn=0) for side in (24, 30, 36)}
    for page, pred in zip(report['pages'], predictions):
        with np.load(caches[page['cache_sha256']]) as data:
            boxes, x = data['boxes'], data['features']
        gray = cv2.imread(str(ROOT / 'datasets' / pred['image_path']), 0)
        adjusted, valid = transform(boxes, refiner.predict(x), gray.shape[1], gray.shape[0])
        gt = np.asarray(pred['gt']).reshape(-1, 4)
        confidence = verifier.decision_function(x)
        overlap = ious(adjusted, gt)
        gated = (confidence >= 1.25) & valid
        matched = {m['ground_truth_index'] for m in pred['variants']['tight_refined_1.25']['matches']}
        for side in (24, 30, 36):
            keep = gated & ((boxes[:, 2] - boxes[:, 0]) >= side)
            ids = nms(adjusted[keep], confidence[keep], .3)
            result = match_detections(adjusted[keep][ids], confidence[keep][ids], gt)
            if side == 36:
                assert all(result[k] == page['results']['tight_refined_1.25'][k] for k in ('tp', 'fp', 'fn'))
                assert result['matches'] == pred['variants']['tight_refined_1.25']['matches']
            for key in ('tp', 'fp', 'fn'):
                totals[str(side)][key] += result[key]
        for index, face in enumerate(gt):
            available = (overlap[:, index] >= .5) & valid
            if index in matched:
                reason = 'matched_TP'
            elif not available.any():
                reason = 'no_refined_IoU50_candidate'
            elif not (available & gated).any():
                reason = 'verifier_filtered'
            elif not (available & gated & ((boxes[:, 2] - boxes[:, 0]) >= 36)).any():
                reason = 'minimum_side_filtered'
            else:
                reason = 'final_NMS_or_matching_competition'
            details.append({'image_id': page['image_id'], 'annotation_index': index, 'gt': face.tolist(),
                            'reason': reason, 'gt_max_side': float(max(face[2:] - face[:2])),
                            'best_refined_IoU': float(overlap[:, index].max()) if len(boxes) else 0,
                            'best_margin_IoU50': float(confidence[available].max()) if available.any() else None})
        print(json.dumps({'image_id': page['image_id'], 'completed_faces': len(details)}), flush=True)
    for value in totals.values():
        value['precision'] = value['tp'] / max(1, value['tp'] + value['fp'])
        value['recall'] = value['tp'] / max(1, value['tp'] + value['fn'])
        value['f1'] = 2 * value['tp'] / max(1, 2 * value['tp'] + value['fp'] + value['fn'])
    assert totals['36'] == report['summary']['tight_refined_1.25']
    output = {'reasons': dict(Counter(r['reason'] for r in details)), 'faces': details,
              'minimum_side_controls': totals, 'threshold': 1.25, 'final_nms': .3,
              'source_sha256': sha(source), 'predictions_sha256': sha(prediction_path),
              'script_sha256': sha(Path(__file__)),
              'scope': 'Explored validation20,252GT; actual baseline one-to-one matches reproduced; availability attribution is ordered and not causal proof. No training data generated from validation.'}
    (ROOT / 'results/cascade_auto_expanded_refiner_diagnosis20.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'reasons': output['reasons'], 'minimum_side_controls': totals}), flush=True)

if __name__ == '__main__':
    main()
