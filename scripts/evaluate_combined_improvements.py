"""Compare verifier and box refinement on the same cached search128 proposals."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compare_multiscale import choose_pages
from src.multiscale import nms
from src.detection_metrics import match_detections
from evaluate_box_refiner import transform


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cascade', type=Path, default=ROOT/'datasets/derived/step8_cascade_search128/model.json')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--reference', type=Path, default=ROOT/'results/step8_cascade_search128_validation.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/step8_combined_improvements_validation.json')
    parser.add_argument('--prediction-dir', type=Path, default=ROOT/'datasets/derived/step8_combined_improvements')
    parser.add_argument('--pages', type=int, default=8)
    parser.add_argument('--pages-per-work', type=int, default=1)
    parser.add_argument('--tight-verifier',type=Path,default=ROOT/'datasets/derived/step8_tight_crop_verifier/model.joblib')
    parser.add_argument('--extra-thresholds',type=float,nargs='*',default=[])
    parser.add_argument('--first-nms',type=float,default=.3)
    parser.add_argument('--refiner',type=Path,default=ROOT/'models/box_refiner_ridge_v1.joblib')
    args = parser.parse_args()
    if not np.isfinite(args.first_nms) or not 0<=args.first_nms<=1:
        parser.error('first NMS must be finite and in [0,1]')
    cascade = args.cascade.resolve()
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    directory = args.cache_dir or cascade.parent/'validation_cache'
    state = {p.name:digest(p) for p in (ROOT/'src').glob('*.py')}
    if json.loads((directory/'source_state.json').read_text()) != state:
        raise ValueError('cache source changed')
    paths = {'old':ROOT/'models/candidate_hog_rbf_step3_v1.joblib',
             'tight':args.tight_verifier.resolve()}
    models = {k:joblib.load(p) for k,p in paths.items()}
    refiner_path = args.refiner.resolve()
    refiner = joblib.load(refiner_path)
    if not all(np.isfinite(t) for t in args.extra_thresholds):
        parser.error('thresholds must be finite')
    thresholds = tuple(sorted({1.5,1.75,*args.extra_thresholds}))
    totals = {f'{v}_{mode}_{t:g}':dict(tp=0,fp=0,fn=0) for v in models for mode in ('raw','postnms','refined') for t in thresholds}
    pages, predictions = [], []
    for page in choose_pages(manifest,args.pages,args.pages_per_work):
        identity = ':'.join((page['image_id'],digest(cascade),digest(manifest),str(args.first_nms),
                             digest(ROOT/'src/candidate_verifier.py'),digest(ROOT/'src/multiscale.py'),
                             digest(ROOT/'datasets'/page['image_path'])))
        cache = directory/(hashlib.sha256(identity.encode()).hexdigest()+'.npz')
        with np.load(cache) as data:
            boxes,scores,x = data['boxes'],data['scores'],data['features']
        gray = cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        gt = np.asarray([a['bbox'] for a in page['annotations']])
        adjusted,valid = transform(boxes,refiner.predict(x),gray.shape[1],gray.shape[0])
        item = {'image_id':page['image_id'],'cache_sha256':digest(cache),'results':{}}
        pred = {'image_id':page['image_id'],'image_path':page['image_path'],'gt':gt.tolist(),'variants':{}}
        for name,model in models.items():
            confidence = model.decision_function(x)
            for t in thresholds:
                keep = (confidence>=t)&((boxes[:,2]-boxes[:,0])>=36)
                for mode in ('raw','postnms','refined'):
                    selected = keep&valid if mode=='refined' else keep
                    bb = adjusted[selected] if mode=='refined' else boxes[selected]
                    ss = scores[selected] if mode=='raw' else confidence[selected]
                    ids = np.arange(len(bb)) if mode=='raw' else nms(bb,ss,.3)
                    result = match_detections(bb[ids],ss[ids],gt)
                    key = f'{name}_{mode}_{t:g}'
                    item['results'][key] = {k:result[k] for k in ('tp','fp','fn')}
                    pred['variants'][key] = {'boxes':bb[ids].tolist(),'scores':ss[ids].tolist(),'matches':result['matches']}
                    for k in ('tp','fp','fn'):
                        totals[key][k] += result[k]
        pages.append(item)
        predictions.append(pred)
        print(item,flush=True)
    for v in totals.values():
        v['precision']=v['tp']/max(1,v['tp']+v['fp'])
        v['recall']=v['tp']/max(1,v['tp']+v['fn'])
        v['f1']=2*v['tp']/max(1,2*v['tp']+v['fp']+v['fn'])
    # Verify original verifier/reference against the previous ordinary evaluator.
    reference = json.loads(args.reference.read_text())
    if reference['cascade_sha256'] != digest(cascade):
        raise ValueError('reference cascade hash mismatch')
    if reference['nms_iou'] != args.first_nms:
        raise ValueError('reference first NMS mismatch')
    if [p['image_id'] for p in pages] != [p['id'] for p in reference['pages']]:
        raise ValueError('reference page identity/order mismatch')
    for t in (1.5,1.75):
        if totals[f'old_raw_{t:g}'] != reference['summary'][f'rbf{t:g}_side36']:
            raise ValueError('cached baseline does not match reference')
    out = args.prediction_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    report = {'split':'validation','selection':f'fixed step5-comparison SHA-256 order; round robin {args.pages} pages, max {args.pages_per_work} per work',
              'step':2,'scale_factor':1.2,'first_nms':args.first_nms,'final_nms':.3,'thresholds':thresholds,
              'timing':'reused cached proposals/features; deployment latency not measured',
              'refiner_training_limitation':'refiner trained on old Cascade proposals; transfer to specified Cascade evaluated here' if refiner_path==ROOT/'models/box_refiner_ridge_v1.joblib' else 'custom train-only refiner; consult its provenance for sampling scope; transfer evaluated here',
              'sha256':{str(p.relative_to(ROOT)):digest(p) for p in (cascade,manifest,refiner_path,*paths.values())},
              'pages':pages,'summary':totals}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(totals)


if __name__=='__main__':
    main()
