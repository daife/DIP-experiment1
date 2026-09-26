"""Evaluate fixed baseline and two combined configurations on new validation pages."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compare_multiscale import choose_pages
from src.cascade import cascade_from_dict
from src.multiscale import PyramidConfig,detect_multiscale,nms
from src.candidate_verifier import features
from src.detection_metrics import match_detections
from evaluate_box_refiner import transform


def main():
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    old_pages={p['image_id'] for p in choose_pages(manifest,8)}
    rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    eligible=[p for p in rows if p['split']=='validation' and p['annotations'] and p['image_id'] not in old_pages]
    eligible.sort(key=lambda p:hashlib.sha256(('step8-validation-expansion:'+p['image_id']).encode()).digest())
    selected=[];groups=set()
    for p in eligible:
        if p['source_group'] not in groups:
            selected.append(p);groups.add(p['source_group'])
    paths={'original_cascade':ROOT/'models/step4_cascade.json',
           'search_cascade':ROOT/'datasets/derived/step8_cascade_search128/model.json',
           'old_verifier':ROOT/'models/candidate_hog_rbf_step3_v1.joblib',
           'tight_verifier':ROOT/'datasets/derived/step8_tight_crop_verifier/model.joblib',
           'refiner':ROOT/'models/box_refiner_ridge_v1.joblib'}
    cascades={k:cascade_from_dict(json.loads(paths[k].read_text())) for k in ('original_cascade','search_cascade')}
    models={k:joblib.load(paths[k]) for k in ('old_verifier','tight_verifier','refiner')}
    totals={k:dict(tp=0,fp=0,fn=0) for k in ('original','search_tight_refined1.5','search_old_refined1.75')}
    pages=[];predictions=[]
    for page in selected:
        gray=cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        gt=np.asarray([a['bbox'] for a in page['annotations']])
        item={'image_id':page['image_id'],'source_group':page['source_group'],'gt':len(gt),'results':{},'seconds':{}}
        pred={'image_id':page['image_id'],'image_path':page['image_path'],'gt':gt.tolist(),'variants':{}}
        for cascade_name in cascades:
            start=perf_counter()
            boxes,scores,_=detect_multiscale(gray,cascades[cascade_name],PyramidConfig(1.2,2,.3))
            x=features(gray,boxes)
            variants=[('original','old_verifier',1.5,False)] if cascade_name=='original_cascade' else [('search_tight_refined1.5','tight_verifier',1.5,True),('search_old_refined1.75','old_verifier',1.75,True)]
            for name,verifier,t,refine in variants:
                confidence=models[verifier].decision_function(x)
                keep=(confidence>=t)&((boxes[:,2]-boxes[:,0])>=36)
                bb,ss=boxes[keep],scores[keep]
                if refine and len(bb):
                    bb,valid=transform(bb,models['refiner'].predict(x[keep]),gray.shape[1],gray.shape[0])
                    bb,ss=bb[valid],confidence[keep][valid]
                    ids=nms(bb,ss,.3)
                    bb,ss=bb[ids],ss[ids]
                result=match_detections(bb,ss,gt)
                item['results'][name]={k:result[k] for k in ('tp','fp','fn')}
                pred['variants'][name]={'boxes':bb.tolist(),'scores':ss.tolist(),'matches':result['matches']}
                for k in ('tp','fp','fn'):totals[name][k]+=result[k]
            item['seconds'][cascade_name]=perf_counter()-start
        pages.append(item);predictions.append(pred);print(item,flush=True)
    for v in totals.values():
        v['precision']=v['tp']/max(1,v['tp']+v['fp']);v['recall']=v['tp']/max(1,v['tp']+v['fn']);v['f1']=2*v['tp']/max(1,2*v['tp']+v['fp']+v['fn'])
    out=ROOT/'datasets/derived/step8_validation_expansion';out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    report={'split':'validation','selection':'SHA-256 step8-validation-expansion:image_id; exclude initial eight pages; one page per distinct validation source group',
            'configurations_fixed_before_expansion':True,'pages':pages,'summary':totals,
            'timing_scope':'original includes one verifier; search includes two verifiers/refined variants; not directly comparable latency',
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,*paths.values())}}
    (ROOT/'results/step8_validation_expansion.json').write_text(json.dumps(report,indent=2)+'\n');print(totals)


if __name__=='__main__':main()
