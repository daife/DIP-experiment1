"""Evaluate an exported detector configuration including real landmark inference."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.detector import AnimeFaceDetector
from src.detection_metrics import match_detections
from compare_multiscale import choose_pages
from evaluate_candidate_verifier import choose_test_pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--pages',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    selected=choose_pages(manifest,args.pages) if args.split=='validation' else choose_test_pages(manifest,args.pages)
    detector=AnimeFaceDetector(ROOT/'models',config_name=args.config)
    totals=dict(tp=0,fp=0,fn=0);pages=[]
    for page in selected:
        image=cv2.imread(str(ROOT/'datasets'/page['image_path']))
        start=perf_counter();detections=detector.detect(image);seconds=perf_counter()-start
        boxes=np.asarray([d['bbox'] for d in detections]).reshape(-1,4)
        scores=np.asarray([d['score'] for d in detections])
        if (np.diff(scores)>0).any() or any(np.asarray(d['landmarks']).shape!=(28,2) or not np.isfinite(d['landmarks']).all() for d in detections):
            raise ValueError('invalid public detector score order or landmarks')
        gt=np.asarray([a['bbox'] for a in page['annotations']])
        result=match_detections(boxes,scores,gt)
        item={'image_id':page['image_id'],'seconds':seconds,'results':{k:result[k] for k in ('tp','fp','fn')},'detections':detections}
        for k in totals:totals[k]+=result[k]
        pages.append(item);print({k:v for k,v in item.items() if k!='detections'},flush=True)
    totals['precision']=totals['tp']/max(1,totals['tp']+totals['fp']);totals['recall']=totals['tp']/max(1,totals['tp']+totals['fn']);totals['f1']=2*totals['tp']/max(1,2*totals['tp']+totals['fp']+totals['fn'])
    config_path=ROOT/'models'/args.config
    inputs=[config_path,manifest,ROOT/'models'/detector.config['cascade'],ROOT/'models'/detector.config['landmark_model'],ROOT/'models'/detector.config['candidate_verifier']['model']]
    if 'box_refiner' in detector.config:inputs.append(ROOT/'models'/detector.config['box_refiner']['model'])
    report={'split':args.split,'config':args.config,'selection':('step5-comparison' if args.split=='validation' else 'step8-test')+' SHA-256 order; distinct groups',
            'not_blind':True,'iou_threshold':.5,'summary':totals,'pages':pages,'mean_seconds':sum(p['seconds'] for p in pages)/len(pages),
            'timing_scope':'full public detect including landmark inference; no cache',
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(totals)


if __name__=='__main__':main()
