"""Veto high-margin background windows with a local-context face teacher.

Local crops address small faces missed by whole-page downscaling. Teacher
absence is still not proof of absence; original data remain immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from anime_face_detector import get_checkpoint_path
from anime_face_detector._face import load_face_detector
from prepare_detection_dataset import near_face

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,default=ROOT/'datasets/derived/cascade_auto_background_merged_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1')
    parser.add_argument('--all-contexts',action='store_true')
    parser.add_argument('--preceding-guard',type=Path)
    parser.add_argument('--quarantine-json',type=Path)
    args=parser.parse_args()
    pool=args.pool;output=args.output
    output.mkdir(parents=True,exist_ok=False)
    rows=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
    mining=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v2/cascade_auto_background_merged_v1_mining.npz'
    if args.all_contexts:
        selected=set(range(len(rows)))
    else:
        with np.load(mining) as data:
            selected=set(data['selected_indices'].tolist())
    allowed=set(range(len(rows)))
    if args.preceding_guard:
        parent=json.loads((args.preceding_guard/'provenance.json').read_text())
        summary=json.loads((args.preceding_guard/'summary.json').read_text())
        assert parent['source_sha256']==sha(pool/'candidates.jsonl')
        assert summary['accepted_indices_sha256']==sha(args.preceding_guard/'accepted_indices.npy')
        indices=np.load(args.preceding_guard/'accepted_indices.npy')
        assert len(indices)==len(set(indices.tolist())) and np.all((indices>=0)&(indices<len(rows)))
        allowed=set(indices.tolist())
    # Known face confirmed by agent inspection of source context, never relabelled.
    quarantined={'manga109:KimiHaBokuNoTaiyouDa:092'}
    if args.quarantine_json:
        extra=json.loads(args.quarantine_json.read_text())
        quarantined.update(extra.get('quarantined_pages',extra.get('quarantine_pages',[])))
    torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoint=get_checkpoint_path('yolov3');model=load_face_detector('yolov3',checkpoint,'cpu')
    provenance={'source_sha256':sha(pool/'candidates.jsonl'),'mining_sha256':sha(mining) if not args.all_contexts else None,
        'preceding_guard_sha256':sha(args.preceding_guard/'provenance.json') if args.preceding_guard else None,
        'all_contexts':args.all_contexts,'quarantine_sha256':sha(args.quarantine_json) if args.quarantine_json else None,
        'script_sha256':sha(Path(__file__)),'teacher_sha256':sha(checkpoint),'teacher':'yolov3',
        'threshold':.1,'context_rule':'centered square max(192,3*candidate_side), clipped to image; selected hard background candidates only',
        'quarantined_pages':sorted(quarantined),'limitation':'Unselected backgrounds keep preceding whole-page teacher decision; local teacher misses still possible; no human approval.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    decisions=[];page_id=None;image=None
    for index,row in enumerate(rows):
        reason=None;context=None;detections=[]
        if row['page_id'] in quarantined:
            reason='confirmed_face_source_page_quarantine'
        elif index not in allowed:
            reason='preceding_teacher_guard_rejected'
        elif index in selected:
            if row['page_id']!=page_id:
                image=cv2.imread(str(ROOT/'datasets'/row['parent_image_path']));page_id=row['page_id']
            if image is None:
                raise ValueError(row['parent_image_path'])
            box=row['bbox_xyxy'];side=max(box[2]-box[0],box[3]-box[1]);cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            half=max(192,3*side)/2
            context=[max(0,int(cx-half)),max(0,int(cy-half)),min(image.shape[1],int(np.ceil(cx+half))),min(image.shape[0],int(np.ceil(cy+half)))]
            x1,y1,x2,y2=context
            for detection in model.detect(image[y1:y2,x1:x2]):
                if detection[4]>=.1:
                    detections.append([float(detection[0]+x1),float(detection[1]+y1),float(detection[2]+x1),float(detection[3]+y1),float(detection[4])])
            if near_face(box,[d[:4] for d in detections]):
                reason='local_teacher_face_overlap'
        decision={'array_index':index,'id':row['id'],'page_id':row['page_id'],'context':context,
                  'teacher_detections':detections,'reason':reason,'decision':'reject' if reason else 'keep_auto'}
        decisions.append(decision)
        with (output/'decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(decision)+'\n')
        if index%100==0:
            print({'checked':index+1,'total':len(rows),'rejected':sum(r['decision']=='reject' for r in decisions)},flush=True)
    accepted=np.asarray([r['array_index'] for r in decisions if r['decision']=='keep_auto'],dtype=np.int64)
    np.save(output/'accepted_indices.npy',accepted)
    report={'total':len(rows),'accepted':len(accepted),'rejected':len(rows)-len(accepted),
            'selected_hard_contexts':len(selected),'accepted_indices_sha256':sha(output/'accepted_indices.npy'),
            'decisions_sha256':sha(output/'decisions.jsonl')}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
