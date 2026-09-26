"""Compare verifier and box refinement on the same cached search128 proposals."""
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
    cascade = ROOT/'datasets/derived/step8_cascade_search128/model.json'
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    directory = cascade.parent/'validation_cache'
    state = {p.name:digest(p) for p in (ROOT/'src').glob('*.py')}
    if json.loads((directory/'source_state.json').read_text()) != state:
        raise ValueError('cache source changed')
    paths = {'old':ROOT/'models/candidate_hog_rbf_step3_v1.joblib',
             'tight':ROOT/'datasets/derived/step8_tight_crop_verifier/model.joblib'}
    models = {k:joblib.load(p) for k,p in paths.items()}
    refiner_path = ROOT/'models/box_refiner_ridge_v1.joblib'
    refiner = joblib.load(refiner_path)
    thresholds = (1.5,1.75)
    totals = {f'{v}_{mode}_{t:g}':dict(tp=0,fp=0,fn=0) for v in models for mode in ('raw','postnms','refined') for t in thresholds}
    pages, predictions = [], []
    for page in choose_pages(manifest,8):
        identity = ':'.join((page['image_id'],digest(cascade),digest(manifest),'0.3',
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
    reference = json.loads((ROOT/'results/step8_cascade_search128_validation.json').read_text())
    for t in thresholds:
        if totals[f'old_raw_{t:g}'] != reference['summary'][f'rbf{t:g}_side36']:
            raise ValueError('cached baseline does not match reference')
    out = ROOT/'datasets/derived/step8_combined_improvements'
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    report = {'split':'validation','selection':'fixed step5-comparison SHA-256 order; eight distinct groups',
              'step':2,'scale_factor':1.2,'first_nms':.3,'final_nms':.3,'thresholds':thresholds,
              'timing':'reused cached proposals/features; deployment latency not measured',
              'refiner_training_limitation':'refiner trained on old Cascade proposals; transfer to search128 evaluated here',
              'sha256':{str(p.relative_to(ROOT)):digest(p) for p in (cascade,manifest,refiner_path,*paths.values())},
              'pages':pages,'summary':totals}
    (ROOT/'results/step8_combined_improvements_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(totals)


if __name__=='__main__':
    main()
