"""Audit exact training windows, including approved legacy hard negatives."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_cascade import load_reviewed_data
from src.cascade import cascade_from_dict, stage_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in args.model:
        payload = json.loads(path.read_text(encoding='utf-8'))
        training = payload['training']
        if training.get('extra_approved_negative_ids') or training.get('hard_negative_repeat',1) != 1:
            raise ValueError('this audit handles unrepeated legacy hard negatives only')
        dataset = ROOT / training['dataset']
        x, y, _ = load_reviewed_data(dataset)['train']
        hard_dir = ROOT / training['hard_negative_dir']
        candidates = [json.loads(s) for s in (hard_dir/'candidates.jsonl').read_text().splitlines()]
        lookup = {r['id']:i for i,r in enumerate(candidates)}
        ids = training['approved_hard_negative_ids']
        if any(candidates[lookup[i]]['split']!='train' or candidates[lookup[i]]['label']!=0 for i in ids):
            raise ValueError('invalid hard negative split/label')
        hard = np.load(hard_dir/'channels11.npy',mmap_mode='r')[[lookup[i] for i in ids]]
        x = np.concatenate((x,hard))
        y = np.concatenate((y,np.zeros(len(hard),dtype=y.dtype)))
        reports.append({'model':str(path), 'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'train_manifest_sha256':hashlib.sha256((dataset/'usable_samples.jsonl').read_bytes()).hexdigest(),
                        'hard_cache_sha256':hashlib.sha256((hard_dir/'channels11.npy').read_bytes()).hexdigest(),
                        'samples':len(y),'hard_negative_count':len(hard),
                        'stages':stage_statistics(cascade_from_dict(payload),x,y)})
    args.output.write_text(json.dumps(reports,indent=2)+'\n')
    print([(r['model'],r['samples'],r['stages'][-1]['positive_passing'],r['stages'][-1]['negative_passing']) for r in reports])


if __name__ == '__main__':
    main()
