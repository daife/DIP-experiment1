"""Compare Cascade retention targets on fixed annotation-derived validation faces.

Only thresholds change; full-page evaluation must follow each saved model.
Calibration faces are explored validation data, not an independent benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cascade import Cascade,CascadeStage,cascade_from_dict,cascade_to_dict,threshold_for_recall,stage_statistics
from src.channels11 import compute_11_channels
from train_cascade import load_reviewed_data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'datasets/derived/cascade_auto_mined_128trees')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/derived/cascade_auto_recalibration')
    p.add_argument('--targets',type=float,nargs='+',default=[.98,.95,.9])
    args=p.parse_args()
    if any(not 0<t<=1 for t in args.targets):
        p.error('targets must be in (0,1]')
    args.output.mkdir(parents=True,exist_ok=True)
    model_path=args.source/'model.json'
    original=cascade_from_dict(json.loads(model_path.read_text()))
    references=json.loads((args.source/'calibration.json').read_text())
    pages={r['image_id']:r for r in map(json.loads,(ROOT/'datasets/manifests/normalized/manga109_faces.jsonl').read_text(encoding='utf-8').splitlines())}
    windows=[]
    loaded={}
    for row in references:
        page=pages[row['page_id']]
        assert row['split']==page['split']=='validation'
        if row['page_id'] not in loaded:
            loaded[row['page_id']]=cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        gray=loaded[row['page_id']]
        if gray is None:
            raise ValueError(page['image_path'])
        x1,y1,x2,y2=row['bbox_xyxy']
        crop=cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
        windows.append(np.stack(compute_11_channels(crop)))
    windows=np.stack(windows)
    old=load_reviewed_data(ROOT/'datasets/derived/step3_detection_v1')['validation']
    reports={}
    for target in args.targets:
        stages=[];alive=np.ones(len(windows),dtype=bool)
        for stage in original.stages:
            threshold=threshold_for_recall(stage.scores(windows[alive]),target)
            updated=CascadeStage(stage.trees,threshold,stage.learning_rate)
            stages.append(updated)
            selected=np.flatnonzero(alive)
            alive[selected]=updated.scores(windows[selected])>=threshold
        model=Cascade(tuple(stages));payload=cascade_to_dict(model)
        payload['calibration']={'source_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                'source_calibration_sha256':hashlib.sha256((args.source/'calibration.json').read_bytes()).hexdigest(),
                                'per_stage_recall_target':target,'only_thresholds_changed':True,'not_blind':True,
                                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        name=f'recall{target:g}'
        (args.output/f'{name}_model.json').write_text(json.dumps(payload,indent=2)+'\n')
        reports[name]={'calibration_retained':int(alive.sum()),'calibration_total':len(windows),
                       'thresholds':[s.threshold for s in stages],
                       'historical_validation_crops':stage_statistics(model,old[0],old[1])}
    (args.output/'stats.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps({k:{'calibration_retained':v['calibration_retained'],'old_positive':v['historical_validation_crops'][-1]['positive_passing'],
                           'old_negative':v['historical_validation_crops'][-1]['negative_passing']} for k,v in reports.items()}))


if __name__=='__main__':
    main()
