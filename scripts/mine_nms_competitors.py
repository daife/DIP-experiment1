"""Mine train-only nonmatching NMS winners that suppress IoU50 face windows."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cascade import cascade_from_dict
from src.channels11 import compute_11_channels
from src.multiscale import detect_multiscale,PyramidConfig
from train_candidate_verifier import choose_pages
from train_box_refiner import ious


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'datasets/derived/cascade_auto_nms_competitors_v1'
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    model=ROOT/'datasets/derived/cascade_auto_scan_positive_localization_v1/model.json'
    quarantine_path=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1/provenance.json'
    quarantine=set(json.loads(quarantine_path.read_text())['quarantined_pages'])
    pages=[p for p in choose_pages(manifest,81) if p['image_id'] not in quarantine]
    selection={'page_ids':[p['image_id'] for p in pages],'manifest_sha256':sha(manifest),'cascade_sha256':sha(model),
        'quarantine_sha256':sha(quarantine_path),'script_sha256':sha(Path(__file__)),
        'scale_factor':1.2,'step':2,'nms_iou':.3,'negative_iou_range':[.1,.5], 'maximum_gt_coverage':.8,
        'per_face':2,'per_page':40,'seed':'none; fixed SHA page order and stable descending Cascade score',
        'limitation':'Localization negatives can contain real face parts; not face-free background labels or manually reviewed data.'}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():
        assert json.loads((output/'selection.json').read_text())==selection
    else:
        (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    cascade=cascade_from_dict(json.loads(model.read_text()))
    progress=[]
    for number,page in enumerate(pages):
        path=output/f'page_{number:03d}.json';shard=path.with_suffix('.npz')
        source=ROOT/'datasets'/page['image_path']
        assert page['split']=='train'
        if path.exists() and shard.exists():
            entry=json.loads(path.read_text());assert entry['image_id']==page['image_id']
            assert entry['image_sha256']==sha(source) and entry['shard_sha256']==sha(shard)
        else:
            gray=cv2.imread(str(source),0)
            if gray is None:
                raise ValueError(source)
            gt=np.asarray([a['bbox'] for a in page['annotations']])
            boxes,scores,_,raw=detect_multiscale(gray,cascade,PyramidConfig(1.2,2,.3),return_pre_nms=True)
            overlap=ious(boxes,gt);raw_overlap=ious(raw['boxes'],gt)
            tl=np.maximum(boxes[:,None,:2],gt[None,:,:2]);br=np.minimum(boxes[:,None,2:],gt[None,:,2:])
            coverage=np.prod(np.maximum(0,br-tl),axis=2)/np.prod(gt[:,2:]-gt[:,:2],axis=1)
            eligible=np.flatnonzero((overlap.max(axis=1)>=.1)&(overlap.max(axis=1)<.5)&(coverage.max(axis=1)<=.8))
            eligible=eligible[np.argsort(-scores[eligible],kind='stable')]
            selected={}
            lost=0
            for gi in range(len(gt)):
                good=np.flatnonzero(raw_overlap[:,gi]>=.5)
                if not len(good):
                    continue
                if np.any(overlap[:,gi]>=.5):
                    continue
                lost+=1
                good=good[np.argsort(-raw['scores'][good],kind='stable')[:8]]
                count=0
                for index in eligible:
                    relations=ious(boxes[index:index+1],raw['boxes'][good])[0]
                    matches=np.flatnonzero((relations>.3)&(scores[index]>=raw['scores'][good]))
                    if not len(matches):
                        continue
                    witness=int(good[matches[0]])
                    selected.setdefault(int(index),[]).append({'annotation_index':gi,'positive_bbox':raw['boxes'][witness].tolist(),
                        'positive_GT_IoU':float(raw_overlap[witness,gi]),'positive_score':float(raw['scores'][witness]),
                        'competitor_IoU':float(relations[matches[0]])})
                    count+=1
                    if count==2:
                        break
            indices=sorted(selected,key=lambda i:(-scores[i],i))[:40]
            rows=[];channels=[]
            for index in indices:
                box=boxes[index].tolist();x1,y1,x2,y2=box
                crop=cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                channels.append(np.stack(compute_11_channels(crop)))
                rows.append({'id':f'nms_competitor:{number:03d}:{len(rows):03d}','page_id':page['image_id'],
                    'source_group':page['source_group'],'parent_image_path':page['image_path'],'split':'train','label':0,
                    'label_basis':'nonmatching_NMS_competitor','bbox_xyxy':box,'max_annotation_IoU':float(overlap[index].max()),
                    'max_gt_coverage':float(coverage[index].max()),'cascade_score':float(scores[index]),
                    'witnesses':selected[index],'crop_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),'review_decision':'unreviewed'})
            np.savez_compressed(shard,channels=np.asarray(channels,dtype=np.uint8).reshape(-1,11,24,24))
            entry={'image_id':page['image_id'],'image_sha256':sha(source),'shard_sha256':sha(shard),
                'GT':len(gt),'NMS_lost_GT':lost,'pre_NMS':len(raw['boxes']),'post_NMS':len(boxes),'rows':rows}
            path.write_text(json.dumps(entry,indent=2)+'\n')
        progress.append({k:v for k,v in entry.items() if k!='rows'}|{'negative':len(entry['rows'])})
        (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress[-1],flush=True)
    rows=[];channels=[];seen=set()
    for number in range(len(pages)):
        entry=json.loads((output/f'page_{number:03d}.json').read_text())
        with np.load(output/f'page_{number:03d}.npz') as data:
            for row,c in zip(entry['rows'],data['channels'],strict=True):
                assert hashlib.sha256(c[0].tobytes()).hexdigest()==row['crop_sha256']
                if row['crop_sha256'] in seen:
                    continue
                seen.add(row['crop_sha256']);rows.append(row);channels.append(c.copy())
    assert rows,'no competitors found'
    np.save(output/'channels11.npy',np.stack(channels))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'provenance.json').write_text(json.dumps({**selection,'negative':len(rows),'pages':progress,
        'candidate_sha256':sha(output/'candidates.jsonl'),'channels_sha256':sha(output/'channels11.npy')},indent=2)+'\n')


if __name__=='__main__':
    main()
