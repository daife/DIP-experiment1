"""Conservatively veto automatic backgrounds using independent face teachers.

Teachers are training-data filters only; they do not replace the Cascade demo.
Source candidates remain immutable/unreviewed. Absence of a teacher detection
is not proof of absence of a face.
"""
import argparse
from collections import defaultdict,Counter
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.1)
    p.add_argument('--suspect-report',type=Path)
    args=p.parse_args()
    if not 0<args.threshold<1:
        p.error('threshold must be in (0,1)')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=list(map(json.loads,(args.dataset/'candidates.jsonl').read_text().splitlines()))
    by_page=defaultdict(list)
    source_pages={r['image_id']:r for r in map(json.loads,(ROOT/'datasets/manifests/normalized/manga109_faces.jsonl').read_text(encoding='utf-8').splitlines())}
    for index,row in enumerate(rows):
        assert row['split']==source_pages[row['page_id']]['split']=='train'
        assert row['label']==0 and row['label_basis']=='zero_overlap_expanded_annotations'
        by_page[row['page_id']].append(index)
    quarantine=set()
    if args.suspect_report:
        suspects=json.loads(args.suspect_report.read_text())['visual_review']['suspect_ids']
        quarantine={r['page_id'] for r in rows if r['id'] in suspects}
        if len(suspects)!=sum(r['id'] in suspects for r in rows):
            raise ValueError('suspect IDs do not match source')
    checkpoints={name:get_checkpoint_path(name) for name in ('yolov3','faster-rcnn')}
    provenance={'source_sha256':digest(args.dataset/'candidates.jsonl'),'threshold':args.threshold,
                'models':{name:{'path':str(path),'sha256':digest(path)} for name,path in checkpoints.items()},
                'script_sha256':digest(Path(__file__)),'quarantine_pages':sorted(quarantine),
                'suspect_report_sha256':digest(args.suspect_report) if args.suspect_report else None,
                'rule':'reject any overlap with teacher box expanded by 12.5% of max side; reject quarantined pages',
                'device':'cpu','torch_threads':2,'human_reviewed':False}
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=args.output/'provenance.json'
    if metadata.exists() and json.loads(metadata.read_text())!=provenance:
        raise ValueError('provenance mismatch; use a new output directory')
    metadata.write_text(json.dumps(provenance,indent=2)+'\n')
    cache_path=args.output/'teacher_pages.jsonl'
    cache={r['page_id']:r for r in map(json.loads,cache_path.read_text().splitlines())} if cache_path.exists() else {}
    models={name:load_face_detector(name,path,'cpu') for name,path in checkpoints.items()}
    accepted,decisions=[],[]
    for page_id,indices in by_page.items():
        if page_id in quarantine:
            boxes=[]
        else:
            if page_id not in cache:
                path=ROOT/'datasets'/rows[indices[0]]['parent_image_path']
                image=cv2.imread(str(path))
                if image is None:
                    raise ValueError(path)
                predictions={name:model.detect(image).tolist() for name,model in models.items()}
                item={'page_id':page_id,'source_image_sha256':digest(path),'predictions':predictions}
                with cache_path.open('a') as stream:
                    stream.write(json.dumps(item)+'\n')
                cache[page_id]=item
            else:
                path=ROOT/'datasets'/rows[indices[0]]['parent_image_path']
                if digest(path)!=cache[page_id]['source_image_sha256']:
                    raise ValueError('cached image changed')
            boxes=[d[:4] for detections in cache[page_id]['predictions'].values() for d in detections if d[4]>=args.threshold]
        for index in indices:
            veto='quarantined_source_page' if page_id in quarantine else ('teacher_face_overlap' if near_face(rows[index]['bbox_xyxy'],boxes) else None)
            decisions.append({'id':rows[index]['id'],'array_index':index,'page_id':page_id,
                              'decision':'reject' if veto else 'keep_auto','reason':veto or 'source_and_teacher_exclusion',
                              'human_reviewed':False})
            if veto is None:
                accepted.append(index)
        print(f'{page_id}: checked {len(decisions)}/{len(rows)}, kept {len(accepted)}',flush=True)
    np.save(args.output/'accepted_indices.npy',np.asarray(sorted(accepted),dtype=np.int64))
    with (args.output/'decisions.jsonl').open('w') as stream:
        for row in sorted(decisions,key=lambda r:r['array_index']):
            stream.write(json.dumps(row)+'\n')
    report={'total':len(rows),'accepted':len(accepted),'rejected':len(rows)-len(accepted),
            'reasons':dict(Counter(r['reason'] for r in decisions)),
            'accepted_indices_sha256':digest(args.output/'accepted_indices.npy'),
            'decisions_sha256':digest(args.output/'decisions.jsonl'),
            'limitation':'Teacher misses remain possible; automatically retained is not human-reviewed.'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
