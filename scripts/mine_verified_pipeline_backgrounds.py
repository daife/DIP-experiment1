"""Mine fresh train backgrounds scored highly by the current Cascade/HOG pair."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cascade import cascade_from_dict
from src.channels11 import compute_11_channels
from src.candidate_verifier import features
from src.multiscale import detect_multiscale,PyramidConfig
from prepare_detection_dataset import near_face


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1'
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    cascade_path=ROOT/'datasets/derived/cascade_auto_scan_positive_localization_v1/model.json'
    verifier_path=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v4/model.joblib'
    prior=ROOT/'datasets/derived/cascade_auto_scan_positives_v1/selection.json'
    excluded=set(json.loads(prior.read_text())['page_ids'])
    quarantine_paths=[ROOT/'datasets/derived/cascade_auto_background_context_guard_v1/provenance.json',
                      ROOT/'datasets/derived/cascade_auto_windows_scale_v2_guard/provenance.json']
    quarantine=set()
    for path in quarantine_paths:
        metadata=json.loads(path.read_text())
        quarantine.update(metadata.get('quarantined_pages',metadata.get('quarantine_pages',[])))
    groups=defaultdict(list)
    for page in map(json.loads,manifest.read_text().splitlines()):
        if page['split']=='train' and page['annotations'] and page['image_id'] not in excluded|quarantine:
            groups[page['source_group']].append(page)
    pages=[min(group,key=lambda p:hashlib.sha256(('pipeline-bg-v1:'+p['image_id']).encode()).digest()) for name,group in sorted(groups.items())]
    assert len(pages)==81
    spec={'manifest_sha256':sha(manifest),'cascade_sha256':sha(cascade_path),'verifier_sha256':sha(verifier_path),
        'prior_selection_sha256':sha(prior),'quarantine_shas':{str(p.relative_to(ROOT)):sha(p) for p in quarantine_paths},
        'quarantined_pages':sorted(quarantine),'page_ids':[p['image_id'] for p in pages],'script_sha256':sha(Path(__file__)),
        'scale_factor':1.2,'step':2,'nms_iou':.3,'minimum_HOG_margin':.5,'per_page':12,
        'seed':'none; fixed pipeline-bg-v1 SHA page order and stable HOG score order',
        'limitation':'Zero expanded-annotation overlap is not proof of no unlabelled face. Teacher exclusion required before training.'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():
        assert json.loads((output/'selection.json').read_text())==spec
    else:
        (output/'selection.json').write_text(json.dumps(spec,indent=2)+'\n')
    cascade=cascade_from_dict(json.loads(cascade_path.read_text()));verifier=joblib.load(verifier_path)
    progress=[]
    for number,page in enumerate(pages):
        metadata=output/f'page_{number:03d}.json';shard=metadata.with_suffix('.npz')
        source=ROOT/'datasets'/page['image_path']
        if metadata.exists() and shard.exists():
            entry=json.loads(metadata.read_text());assert entry['image_id']==page['image_id']
            assert entry['image_sha256']==sha(source) and entry['shard_sha256']==sha(shard)
        else:
            gray=cv2.imread(str(source),0)
            if gray is None:
                raise ValueError(source)
            boxes,scores,_=detect_multiscale(gray,cascade,PyramidConfig(1.2,2,.3))
            gt=[a['bbox'] for a in page['annotations']]
            safe=np.array([i for i,box in enumerate(boxes) if not near_face(box,gt)],dtype=np.int64)
            vectors=features(gray,boxes[safe])
            margins=verifier.decision_function(vectors) if len(vectors) else np.empty(0)
            ranked=np.argsort(-margins,kind='stable')
            rows=[];channels=[];hogs=[];seen=set()
            for index in ranked:
                if margins[index]<.5 or len(rows)==12:
                    break
                candidate=int(safe[index]);box=boxes[candidate].tolist();x1,y1,x2,y2=box
                crop=cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                digest=hashlib.sha256(crop.tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                channels.append(np.stack(compute_11_channels(crop)));hogs.append(vectors[index])
                rows.append({'id':f'pipeline_background:{number:03d}:{len(rows):03d}','page_id':page['image_id'],
                    'source_group':page['source_group'],'parent_image_path':page['image_path'],'split':'train','label':0,
                    'label_basis':'zero_overlap_expanded_annotations','bbox_xyxy':box,'cascade_score':float(scores[candidate]),
                    'verifier_margin':float(margins[index]),'crop_sha256':digest,'review_decision':'unreviewed'})
            np.savez_compressed(shard,channels=np.asarray(channels,dtype=np.uint8).reshape(-1,11,24,24),
                                features=np.asarray(hogs,dtype=np.float32).reshape(-1,324))
            entry={'image_id':page['image_id'],'image_sha256':sha(source),'shard_sha256':sha(shard),
                'GT':len(gt),'proposals':len(boxes),'safe_background_proposals':len(safe),'rows':rows}
            metadata.write_text(json.dumps(entry,indent=2)+'\n')
        progress.append({k:v for k,v in entry.items() if k!='rows'}|{'negative':len(entry['rows'])})
        (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress[-1],flush=True)
    rows=[];channels=[];hogs=[];seen=set()
    for number in range(len(pages)):
        entry=json.loads((output/f'page_{number:03d}.json').read_text())
        with np.load(output/f'page_{number:03d}.npz') as data:
            for row,c,h in zip(entry['rows'],data['channels'],data['features'],strict=True):
                assert hashlib.sha256(c[0].tobytes()).hexdigest()==row['crop_sha256']
                if row['crop_sha256'] in seen:
                    continue
                seen.add(row['crop_sha256']);rows.append(row);channels.append(c.copy());hogs.append(h.copy())
    assert rows
    np.save(output/'channels11.npy',np.stack(channels));np.save(output/'hog_features.npy',np.stack(hogs))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (output/'provenance.json').write_text(json.dumps({**spec,'negative':len(rows),'pages':progress,
        'candidate_sha256':sha(output/'candidates.jsonl'),'channels_sha256':sha(output/'channels11.npy'),
        'hog_sha256':sha(output/'hog_features.npy')},indent=2)+'\n')


if __name__=='__main__':
    main()
