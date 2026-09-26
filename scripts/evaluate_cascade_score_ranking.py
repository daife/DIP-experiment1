"""Compare NMS scores on identical surviving six-stage windows."""
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.channels11 import compute_11_channels
from src.page_scan import _tree_scores, scan_page
from src.cascade import cascade_from_dict
from src.multiscale import pyramid_sizes, nms
from src.candidate_verifier import CandidateVerifier
from src.detection_metrics import match_detections
from compare_multiscale import choose_pages


def scan_scores(gray, cascade):
    height, width = gray.shape
    channels = compute_11_channels(gray)
    gx, gy = np.meshgrid(np.arange(0,width-23,2), np.arange(0,height-23,2))
    xs, ys = gx.ravel(), gy.ravel()
    cumulative = np.zeros(len(xs))
    prefix = np.zeros(len(xs))
    for si, stage in enumerate(cascade.stages):
        scores = np.zeros(len(xs))
        for tree in stage.trees:
            scores += stage.learning_rate*_tree_scores(channels,tree,xs,ys)
        cumulative += scores
        if si == 2:
            prefix = scores.copy()
        keep = scores >= stage.threshold
        xs, ys, scores, cumulative, prefix = (v[keep] for v in (xs,ys,scores,cumulative,prefix))
        if not len(xs):
            break
    return np.stack((xs,ys,xs+24,ys+24),axis=1).astype(np.int32), {'last':scores,'sum':cumulative,'stage3':prefix}


def main():
    model_path = ROOT / 'datasets/derived/step8_cascade_search128_stage6/model.json'
    verifier_path = ROOT / 'models/candidate_hog_rbf_step3_v1.joblib'
    manifest = ROOT / 'datasets/manifests/normalized/manga109_faces.jsonl'
    cascade = cascade_from_dict(json.loads(model_path.read_text()))
    verifier = CandidateVerifier(verifier_path)
    # Verify scanner alignment on actual channel windows before full-page use.
    sample = np.random.default_rng(20260926).integers(0,256,(64,70),dtype=np.uint8)
    reference_boxes, reference_scores = scan_page(sample,cascade,step=2)
    check_boxes, check_scores = scan_scores(sample,cascade)
    if not np.array_equal(reference_boxes,check_boxes) or not np.array_equal(reference_scores,check_scores['last']):
        raise ValueError('scanner differs from production')
    names = ('last','sum','stage3')
    thresholds = (1.5,1.75)
    totals = {f'{n}_rbf{t:g}':dict(tp=0,fp=0,fn=0) for n in names for t in thresholds}
    pages = []
    for page in choose_pages(manifest,8):
        start = perf_counter()
        gray = cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        height,width = gray.shape
        all_boxes, all_scores = [], {n:[] for n in names}
        for w,h in pyramid_sizes(width,height,1.2):
            layer = gray if (w,h)==(width,height) else cv2.resize(gray,(w,h),interpolation=cv2.INTER_AREA)
            boxes, scores = scan_scores(layer,cascade)
            mapped = np.rint(boxes*np.asarray([width/w,height/h]*2)).astype(np.int32)
            mapped[:,[0,2]] = np.clip(mapped[:,[0,2]],0,width)
            mapped[:,[1,3]] = np.clip(mapped[:,[1,3]],0,height)
            all_boxes.append(mapped)
            for n in names:
                all_scores[n].append(scores[n])
        pool = np.concatenate(all_boxes)
        rankings = {n:np.concatenate(all_scores[n]) for n in names}
        gt = np.asarray([a['bbox'] for a in page['annotations']])
        item = {'image_id':page['image_id'], 'identical_passed_candidates':len(pool),'results':{}}
        for n in names:
            ids = nms(pool,rankings[n],.3)
            boxes = pool[ids]
            scores = rankings[n][ids]
            hog_scores = verifier.scores(gray,boxes)
            for threshold in thresholds:
                keep = (hog_scores>=threshold)&((boxes[:,2]-boxes[:,0])>=36)
                result = match_detections(boxes[keep],scores[keep],gt)
                key = f'{n}_rbf{threshold:g}'
                item['results'][key] = {k:result[k] for k in ('tp','fp','fn')}
                for k in ('tp','fp','fn'):
                    totals[key][k] += result[k]
        item['seconds_all_variants'] = perf_counter()-start
        pages.append(item)
        print(item,flush=True)
    for t in totals.values():
        t['precision']=t['tp']/max(1,t['tp']+t['fp'])
        t['recall']=t['tp']/max(1,t['tp']+t['fn'])
        t['f1']=2*t['tp']/max(1,2*t['tp']+t['fp']+t['fn'])
    report = {'split':'validation','score_definitions':{'last':'sixth stage','sum':'sum of six stage scores','stage3':'third stage scores on six-stage survivors'},
              'step':2,'scale_factor':1.2,'nms_iou':.3,'thresholds':thresholds,'pages':pages,'summary':totals,
              'scanner_check':'production scan_page boxes/last scores identical on seeded 64x70 image',
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_path,verifier_path,manifest)}}
    (ROOT/'results/step8_cascade_score_ranking_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(totals)


if __name__ == '__main__':
    main()
