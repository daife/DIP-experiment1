"""Build annotation-derived nonmatching train windows around faces.

These are localization negatives, not face-free backgrounds: maximum IoU <=.3
and maximum annotated-face coverage <=.6. Never mix them with human review.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.channels11 import compute_11_channels
from prepare_detection_dataset import square_box


def geometry(box,faces):
    faces=np.asarray(faces,dtype=float).reshape(-1,4)
    intersection=np.prod(np.maximum(0,np.minimum(faces[:,2:],box[2:])-np.maximum(faces[:,:2],box[:2])),axis=1)
    areas=np.prod(faces[:,2:]-faces[:,:2],axis=1)
    window_area=(box[2]-box[0])*(box[3]-box[1])
    return float(np.max(intersection/(window_area+areas-intersection),initial=0)),float(np.max(intersection/areas,initial=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/derived/cascade_localization_negatives_v1')
    p.add_argument('--target',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20260926)
    args=p.parse_args()
    if args.target<1:
        p.error('target must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'candidates.jsonl').exists():
        p.error('output already complete; choose a new directory')
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    source=ROOT/'datasets/derived/cascade_auto_v1/samples.jsonl'
    pages={r['image_id']:r for r in map(json.loads,manifest.read_text(encoding='utf-8').splitlines())}
    groups=defaultdict(list)
    for row in map(json.loads,source.read_text(encoding='utf-8').splitlines()):
        if row['label']==1 and not row['horizontal_flip']:
            assert row['split']==pages[row['page_id']]['split']=='train'
            groups[row['page_id']].append(row)
    rng=np.random.default_rng(args.seed)
    ordered=sorted(groups,key=lambda key:hashlib.sha256(f'localization:{args.seed}:{key}'.encode()).digest())
    rows,channels,seen=[],[],set()
    rejected={'geometry':0,'texture':0,'duplicate':0,'bounds':0}
    for page_id in ordered:
        page=pages[page_id]
        gray=cv2.imread(str(ROOT/'datasets'/page['image_path']),0)
        if gray is None:
            raise ValueError(page['image_path'])
        faces=[a['bbox'] for a in page['annotations']]
        for parent in groups[page_id]:
            base=parent['bbox_xyxy']; side=base[2]-base[0]
            cx,cy=(base[0]+base[2])/2,(base[1]+base[3])/2
            accepted=0
            for _ in range(60):
                length=max(24,round(side*float(rng.choice([.75,1.,1.25]))))
                dx,dy=rng.choice([-1.,-.75,-.5,0.,.5,.75,1.],2)
                x,y=round(cx+dx*side-length/2),round(cy+dy*side-length/2)
                box=[x,y,x+length,y+length]
                if min(x,y)<0 or box[2]>gray.shape[1] or box[3]>gray.shape[0]:
                    rejected['bounds']+=1; continue
                iou,coverage=geometry(box,faces)
                if not 0<iou<=.3 or coverage>.6:
                    rejected['geometry']+=1; continue
                crop=cv2.resize(gray[y:y+length,x:x+length],(24,24),interpolation=cv2.INTER_AREA)
                if crop.std()<24 or not .04<=np.mean(crop<220)<=.85:
                    rejected['texture']+=1; continue
                digest=hashlib.sha256(crop.tobytes()).hexdigest()
                if digest in seen:
                    rejected['duplicate']+=1; continue
                seen.add(digest)
                rows.append({'id':f'loc{len(rows):07d}','page_id':page_id,'parent_image_path':page['image_path'],
                             'source_group':page['source_group'],'split':'train','label':0,
                             'review_decision':'unreviewed','label_basis':'nonmatching_localization_window',
                             'bbox_xyxy':box,'parent_positive_id':parent['id'],'crop_sha256':digest,
                             'max_face_iou':iou,'max_face_coverage':coverage})
                channels.append(np.stack(compute_11_channels(crop)))
                accepted+=1
                if accepted==4 or len(rows)==args.target:
                    break
            if len(rows)==args.target:
                break
        if len(rows)==args.target:
            break
        if len(rows)%1000<20:
            print(f'localization negatives: {len(rows)}',flush=True)
    if len(rows)<args.target:
        raise ValueError(f'only {len(rows)} eligible windows, target {args.target}')
    np.save(args.output/'channels11.npy',np.stack(channels))
    with (args.output/'candidates.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    report={'seed':args.seed,'total':len(rows),'pages':len({r['page_id'] for r in rows}),
            'works':len({r['source_group'] for r in rows}),'rejected':rejected,'human_reviewed':False,
            'label_basis':'IoU<=.3 and annotated-face coverage<=.6, nonzero overlap; not face-free background',
            'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'parent_samples_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'candidates_sha256':hashlib.sha256((args.output/'candidates.jsonl').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
