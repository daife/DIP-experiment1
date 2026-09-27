"""Automatic train-only expansion and three reproducible hard-negative rounds.

Automatic annotation-derived labels are never marked human-reviewed. Existing
datasets and deployed models remain intact. Evaluation uses fixed validation works.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_to_dict, fit_cascade, stage_statistics
from src.channels11 import compute_11_channels
from prepare_detection_dataset import near_face, ordered_round_robin, square_box
from train_cascade import load_reviewed_data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build(args, pages):
    rng = np.random.default_rng(args.seed)
    selections = ordered_round_robin(pages, 'automatic-expansion-v1')[:args.faces]
    by_page = {}
    for page, index in selections:
        by_page.setdefault(page['image_id'], []).append(index)
    channels, labels, rows = [], [], []
    audit = {'face_overlap_rejected': 0, 'low_texture_rejected': 0, 'duplicate_rejected': 0}
    ordered = sorted(pages, key=lambda p: hashlib.sha256(f"{args.seed}:{p['image_id']}".encode()).digest())
    negatives = 0
    fingerprints = set()
    for page in ordered:
        if page['image_id'] not in by_page and negatives >= args.negative_pool:
            continue
        gray = cv2.imread(str(ROOT / 'datasets' / page['image_path']), 0)
        if gray is None:
            raise ValueError(page['image_path'])
        faces = [a['bbox'] for a in page['annotations']]
        def append(box, label, annotation=None, flip=False):
            x1, y1, x2, y2 = box
            if min(x1, y1) < 0 or x2 > gray.shape[1] or y2 > gray.shape[0]:
                return False
            crop = cv2.resize(gray[y1:y2, x1:x2], (24, 24), interpolation=cv2.INTER_AREA)
            if flip:
                crop = np.fliplr(crop).copy()
            fingerprint = hashlib.sha256(crop.tobytes()).hexdigest()
            if fingerprint in fingerprints:
                audit['duplicate_rejected'] += 1
                return False
            if label == 0 and (crop.std() < 24 or not .04 <= np.mean(crop < 220) <= .85):
                audit['low_texture_rejected'] += 1
                return False
            fingerprints.add(fingerprint)
            channels.append(np.stack(compute_11_channels(crop)))
            labels.append(label)
            rows.append({'id': f'auto{len(rows):07d}', 'page_id': page['image_id'],
                         'parent_image_path': page['image_path'], 'source_group': page['source_group'],
                         'split': 'train', 'label': label, 'bbox_xyxy': box,
                         'annotation_index': annotation, 'horizontal_flip': flip,
                         'review_decision': 'unreviewed', 'label_basis': 'source_face_annotation' if label else 'zero_overlap_expanded_annotations',
                         'crop_sha256': fingerprint})
            return True
        for index in by_page.get(page['image_id'], []):
            box = square_box(faces[index], margin=1.0)
            append(box, 1, index)
            append(box, 1, index, True)
        for _ in range(240):
            if negatives >= args.negative_pool:
                break
            side = int(rng.choice([32, 48, 64, 96, 128, 192, 256, 320]))
            if side > min(gray.shape):
                continue
            x, y = int(rng.integers(gray.shape[1]-side+1)), int(rng.integers(gray.shape[0]-side+1))
            box = [x, y, x+side, y+side]
            if near_face(box, faces):
                audit['face_overlap_rejected'] += 1
                continue
            if append(box, 0):
                negatives += 1
                if sum(r['page_id'] == page['image_id'] and r['label'] == 0 for r in rows[-32:]) >= 12:
                    break
        if len(rows) % 1000 < 20:
            print(f'build: {len(rows)} samples, {negatives} negatives', flush=True)
    x, y = np.stack(channels), np.asarray(labels, dtype=np.uint8)
    if negatives < args.negative_pool:
        raise ValueError(f'only {negatives} negatives; requested {args.negative_pool}')
    np.save(args.output / 'channels.npy', x)
    np.save(args.output / 'labels.npy', y)
    with (args.output / 'samples.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    save(args.output / 'build_audit.json', {**audit, 'positive': int(y.sum()), 'negative': negatives,
         'unique_positive_annotations': len({(r['page_id'], r['annotation_index']) for r in rows if r['label']}),
         'human_reviewed': False, 'limitation': 'Unannotated faces may remain in automatically labelled backgrounds.'})
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/derived/cascade_auto_v1')
    parser.add_argument('--faces', type=int, default=8000)
    parser.add_argument('--negative-pool', type=int, default=40000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--skip-evaluation', action='store_true')
    args = parser.parse_args()
    if min(args.faces, args.negative_pool, args.rounds) < 1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl'
    pages = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    groups = {}
    for page in pages:
        if groups.setdefault(page['source_group'], page['split']) != page['split']:
            raise ValueError('source group split leakage')
    provenance = {'seed': args.seed, 'manifest_sha256': sha(manifest), 'script_sha256': sha(Path(__file__)),
                  'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    existing = args.output / 'provenance.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8')) != provenance:
        raise ValueError('provenance mismatch; choose a new output directory')
    save(existing, provenance)
    if (args.output / 'build_audit.json').exists():
        x, y = np.load(args.output/'channels.npy', mmap_mode='r'), np.load(args.output/'labels.npy')
    else:
        x, y = build(args, [p for p in pages if p['split'] == 'train'])
    validation = load_reviewed_data(ROOT / 'datasets/derived/step3_detection_v1')['validation']
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(args.seed)
    model = None
    for iteration in range(args.rounds):
        rng = np.random.default_rng(args.seed + iteration)
        model_path = args.output / f'round{iteration+1}_model.json'
        if model_path.exists():
            from src.cascade import cascade_from_dict
            model = cascade_from_dict(json.loads(model_path.read_text(encoding='utf-8')))
        else:
            started = perf_counter()
            if model is None:
                selected_negative = rng.choice(negative, min(20000, len(negative)), replace=False)
                hard = []
            else:
                passed, scores, trace = model.evaluate(x[negative])
                # Score only within the same survival depth, then keep top hard windows
                # plus seeded broad background coverage. No validation/test enters mining.
                depth = np.zeros(len(negative), dtype=int)
                alive = np.ones(len(negative), dtype=bool)
                margin = np.zeros(len(negative))
                for stage in model.stages:
                    idx = np.flatnonzero(alive)
                    margin[idx] = stage.scores(x[negative[idx]]) - stage.threshold
                    alive[idx] = margin[idx] >= 0
                    depth[alive] += 1
                hard = negative[np.lexsort((-margin, -depth))[:min(15000, len(negative))]]
                selected_negative = np.unique(np.concatenate([hard, rng.choice(negative, min(15000, len(negative)), replace=False)]))
            selected = np.concatenate([positive, selected_negative])
            np.save(args.output / f'round{iteration+1}_indices.npy', selected)
            print(f'round {iteration+1}: {len(positive)} positive / {len(selected_negative)} negative; training', flush=True)
            model = fit_cascade(x[selected], y[selected], validation[0], validation[1], seed=args.seed,
                                num_stages=3, num_trees=18, root_candidates=64, child_candidates=32, target_recall=.995)
            payload = cascade_to_dict(model)
            payload['training'] = {**provenance, 'round': iteration+1, 'fully_reviewed': False,
                                   'samples_sha256': sha(args.output/'samples.jsonl'),
                                   'indices_sha256': sha(args.output/f'round{iteration+1}_indices.npy')}
            save(model_path, payload)
            save(args.output/f'round{iteration+1}_stats.json', {'seconds': perf_counter()-started,
                 'training': stage_statistics(model, x[selected], y[selected]),
                 'validation': stage_statistics(model, validation[0], validation[1]),
                 'hard_negative_count': len(hard), 'model_sha256': sha(model_path)})
        if not args.skip_evaluation:
            report = ROOT / f'results/cascade_auto_v1_round{iteration+1}_validation.json'
            if not report.exists():
                subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_candidate_verifier.py'),
                                '--cascade', str(model_path), '--verifier', str(ROOT/'models/candidate_hog_rbf_step3_v1.joblib'),
                                '--output', str(report), '--cache-dir', str(args.output/f'round{iteration+1}_validation_cache')], check=True)
    save(args.output/'finished.json', {'rounds': args.rounds, 'status': 'experiments_finished_not_acceptance',
         'note': 'Deployment requires full-page metric and visual comparison; automatic labels are not human review.'})


if __name__ == '__main__':
    main()
