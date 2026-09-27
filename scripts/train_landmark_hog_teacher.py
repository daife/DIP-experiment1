"""Compare nonlinear global shape initialization on fixed teacher splits."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iterate_landmark_teacher import evaluate, digest
from src.landmark_hog import HogLandmarkRegressor, global_features
from src.landmark_regression import features, fit_stage, make_offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components', type=int, default=512)
    parser.add_argument('--gamma', type=float, nargs='+', default=[.025, .05, .1])
    parser.add_argument('--initializer-alpha', type=float, default=.1)
    parser.add_argument('--run-name', default='landmark_hog_teacher')
    args = parser.parse_args()
    if args.components <= 0 or args.initializer_alpha <= 0 or any(g <= 0 for g in args.gamma):
        raise ValueError('Components, alpha, gamma must be positive')
    if Path(args.run_name).name != args.run_name:
        raise ValueError('Run name must be a filename stem')
    source = ROOT/'datasets/annotations/auto/hrnetv2_full_20260927/annotations.jsonl'
    previous = json.loads((ROOT/'results/landmark_teacher_iterations.json').read_text(encoding='utf-8'))
    if digest(source) != previous['annotation_sha256']:
        raise ValueError('Teacher source has changed')
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    sets = {}
    for split, count in previous['split_counts'].items():
        rows = [r for r in records if r['split'] == split]
        images = np.memmap(ROOT/f'tmp/landmark_iteration/{split}_images.float32',
                           dtype=np.float32, mode='r', shape=(count, 256, 256))
        targets = []
        for r in rows:
            ann = r['annotations'][0]
            box = np.array(ann['bbox'])
            xy = np.array([[p['x'], p['y']] for p in ann['landmarks']])
            targets.append((xy-box[:2])/(box[2:]-box[:2]))
        sets[split] = {'images': images, 'targets': np.array(targets), 'ids': [r['image_id'] for r in rows]}
    train = sets['train']
    x = global_features(train['images'])
    print('HOG features', x.shape, flush=True)
    result = {'annotation_sha256': digest(source), 'seed': 20260927,
              'method': '64px HOG + RBF Nystroem Ridge initial shape + local pixel-difference Ridge',
              'components': args.components, 'initializer_alpha': args.initializer_alpha,
              'truth_policy': 'all teacher points retained; fixed source splits; no human screening',
              'test_status': 'historically explored; not blind', 'trials': []}
    output = ROOT/f'results/{args.run_name}_metrics.json'
    if output.exists():
        raise FileExistsError('Use a new run name to preserve previous results')
    for gamma in args.gamma:
        initializer = make_pipeline(Nystroem(kernel='rbf', gamma=gamma, n_components=args.components, random_state=20260927),
                                    Ridge(alpha=args.initializer_alpha))
        initializer.fit(x, train['targets'].reshape(-1, 56))
        shape = initializer.predict(x).reshape(-1, 28, 2)
        offsets = make_offsets(20260927, 16)
        coefficients = []
        for stage in range(6):
            if stage:
                local = features(train['images'], shape, offsets)
                coef = fit_stage(local, (train['targets']-shape).reshape(-1, 56),
                                 np.ones((len(shape), 28), bool), alpha=10.)
                shape += (np.column_stack([np.ones(len(local)), local])@coef).reshape(-1, 28, 2)
                shape = np.clip(shape, -.25, 1.25)
                coefficients.append(coef)
            if stage not in (0, 3, 5):
                continue
            model = HogLandmarkRegressor(initializer, offsets, np.asarray(coefficients))
            path = ROOT/f'models/{args.run_name}_g{gamma:g}_s{stage}.joblib'
            model.save(path)
            validation = evaluate(HogLandmarkRegressor.load(path), sets['validation'])
            result['trials'].append({'gamma': gamma, 'stages': stage, 'model': path.relative_to(ROOT).as_posix(),
                                     'sha256': digest(path), 'validation': validation})
            print(gamma, stage, validation['teacher_all_point_nme'], validation['median_nme'], flush=True)
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    best = min(result['trials'], key=lambda t: t['validation']['teacher_all_point_nme'])
    result['selected_by_validation'] = best['model']
    result['selected_test'] = evaluate(HogLandmarkRegressor.load(ROOT/best['model']), sets['test'])
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
