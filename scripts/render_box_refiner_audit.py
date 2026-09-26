"""Render before/after proposals around annotated faces on two validation pages."""
import json
import hashlib
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import CandidateVerifier, features
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig, detect_multiscale, nms
from src.detection_metrics import match_detections
from compare_multiscale import choose_pages
from evaluate_box_refiner import transform
from train_box_refiner import ious


def main():
    out = ROOT / 'datasets/derived/step8_box_refiner_audit'
    out.mkdir(parents=True, exist_ok=True)
    cascade = cascade_from_dict(json.loads((ROOT / 'models/step4_cascade.json').read_text()))
    verifier = CandidateVerifier(ROOT / 'models/candidate_hog_rbf_step3_v1.joblib')
    refiner = joblib.load(ROOT / 'models/box_refiner_ridge_v1.joblib')
    report = []
    for page in choose_pages(ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl', 8):
        if not any(name in page['image_id'] for name in ('Tapkun', 'OhWarera')):
            continue
        gray = cv2.imread(str(ROOT / 'datasets' / page['image_path']), 0)
        boxes, _, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2, 2, 0.3))
        x = features(gray, boxes)
        scores = verifier.model.decision_function(x)
        adjusted, valid = transform(boxes, refiner.predict(x), gray.shape[1], gray.shape[0])
        keep = (scores >= 1.5) & ((boxes[:, 2] - boxes[:, 0]) >= 36) & valid
        boxes, adjusted, scores = boxes[keep], adjusted[keep], scores[keep]
        indices = nms(adjusted, scores, 0.3)
        gt = np.asarray([a['bbox'] for a in page['annotations']])
        before = match_detections(boxes, scores, gt)
        after = match_detections(adjusted[indices], scores[indices], gt)
        cells = []
        for gi, target in enumerate(gt):
            center = (target[:2] + target[2:]) / 2
            side = max(target[2:] - target[:2]) * 2.4
            lo = np.maximum(0, np.floor(center - side / 2)).astype(int)
            hi = np.minimum(gray.shape[::-1], np.ceil(center + side / 2)).astype(int)
            crop = cv2.cvtColor(gray[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_GRAY2BGR)
            for proposals, color in ((boxes, (0, 0, 255)), (adjusted[indices], (255, 0, 0)), (gt[gi:gi+1], (0, 180, 0))):
                for b in proposals:
                    if np.all(np.minimum(b[2:], hi) > np.maximum(b[:2], lo)):
                        cv2.rectangle(crop, tuple(b[:2]-lo), tuple(b[2:]-lo), color, 2)
            crop = cv2.resize(crop, (240, 240))
            cell = np.full((275, 240, 3), 255, np.uint8)
            cell[35:] = crop
            bm = any(m['ground_truth_index'] == gi for m in before['matches'])
            am = any(m['ground_truth_index'] == gi for m in after['matches'])
            cv2.putText(cell, f'GT{gi} matched {int(bm)} -> {int(am)}', (5, 23), 0, .5, (0,0,0), 1)
            cells.append(cell)
        sheet = np.full(((len(cells)+3)//4*275, 960, 3), 255, np.uint8)
        for i, cell in enumerate(cells):
            sheet[(i//4)*275:(i//4+1)*275, (i%4)*240:(i%4+1)*240] = cell
        name = page['image_id'].replace(':', '_') + '.jpg'
        if not cv2.imwrite(str(out / name), sheet):
            raise IOError(name)
        report.append({'image_id': page['image_id'], 'source': page['image_path'], 'sheet': name,
                       'before': before, 'after': after, 'original_boxes': boxes.tolist(),
                       'refined_boxes': adjusted[indices].tolist(), 'ground_truth': gt.tolist(),
                       'best_refined_iou_per_gt': ious(adjusted[indices], gt).max(axis=0).tolist(),
                       'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT / 'models/step4_cascade.json',
                                            ROOT / 'models/candidate_hog_rbf_step3_v1.joblib',
                                            ROOT / 'models/box_refiner_ridge_v1.joblib',
                                            ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl')}})
        print(name, flush=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
