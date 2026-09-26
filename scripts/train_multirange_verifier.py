"""Controlled two-margin positive augmentation, with/without reviewed negatives."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.candidate_verifier import features


def main():
    directory = ROOT / 'datasets/derived/step8_multirange_training'
    inputs = []
    for name in ('tight', 'area'):
        path = directory / (name + '.npz')
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['features_sha256'] or meta['split'] != 'train':
            raise ValueError('invalid training cache')
        with np.load(path) as data:
            inputs.append((data['x'], data['y'], meta))
    x1, y1, meta1 = inputs[0]
    x2, y2, meta2 = inputs[1]
    if meta1['manifest_sha256'] != meta2['manifest_sha256'] or not np.array_equal(y1, y2):
        raise ValueError('input examples differ')
    if not np.array_equal(x1[y1==0], x2[y2==0]):
        raise ValueError('background examples changed')
    x = np.concatenate((x1, x2[y2==1]))
    y = np.concatenate((y1, y2[y2==1]))
    review_path = ROOT / 'datasets/annotations/corrected/step8_false_positive_review_merged.csv'
    with review_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len({r['id'] for r in rows}) != len(rows) or any(r['review_decision'] not in ('accept_negative', 'reject') for r in rows):
        raise ValueError('duplicate or unreviewed negative')
    extra = []
    for row in rows:
        if row['review_decision'] == 'reject':
            continue
        if row['split'] != 'train':
            raise ValueError('negative must be from train')
        gray = cv2.imread(str(ROOT / 'datasets' / row['parent_image_path']), 0)
        box = np.asarray(json.loads(row['bbox_xyxy']), dtype=int)
        if gray is None or box.shape != (4,) or (box[:2]<0).any() or box[2]>gray.shape[1] or box[3]>gray.shape[0] or (box[2:]<=box[:2]).any():
            raise ValueError('invalid negative box')
        extra.append(features(gray, box[None])[0])
    summary = []
    variants = [('multirange', x, y),
                ('multirange_hardneg', np.concatenate((x,extra)), np.concatenate((y,np.zeros(len(extra),dtype=y.dtype))))]
    for name, xx, yy in variants:
        model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', cache_size=512, random_state=20260925).fit(xx,yy)
        path = directory / (name + '.joblib')
        joblib.dump(model,path,compress=3)
        report = {'split':'train', 'seed':20260925, 'margins':[1.0,1.15], 'samples':len(yy), 'positive':int(yy.sum()), 'negative':int(len(yy)-yy.sum()),
                  'extra_reviewed_negatives':len(extra) if name.endswith('hardneg') else 0,
                  'rejected_negatives':sum(r['review_decision']=='reject' for r in rows), 'support_vectors':len(model.support_vectors_),
                  'C':1,'gamma':'scale','class_weight':'balanced', 'input_manifest_sha256':meta1['manifest_sha256'],
                  'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'tight.npz',directory/'area.npz',review_path,path)}}
        path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        summary.append(report)
    (ROOT / 'results/step8_multirange_training.json').write_text(json.dumps(summary,indent=2)+'\n')
    print([{k:v for k,v in r.items() if k!='sha256'} for r in summary])


if __name__ == '__main__':
    main()
