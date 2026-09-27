"""Verify automatic background or localization-negative caches and draw 24 samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_detection_dataset import near_face
from build_localization_negatives import geometry

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    pages={r['image_id']:r for r in map(json.loads,(ROOT/'datasets/manifests/normalized/manga109_faces.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=list(map(json.loads,(args.dataset/'candidates.jsonl').read_text().splitlines()))
    cache=np.load(args.dataset/'channels11.npy',mmap_mode='r')
    assert cache.shape==(len(rows),11,24,24) and cache.dtype==np.uint8
    assert len({r['id'] for r in rows})==len(rows)
    positive_ids={r['id'] for r in map(json.loads,(ROOT/'datasets/derived/cascade_auto_v1/samples.jsonl').read_text().splitlines()) if r['label']==1}
    for i,row in enumerate(rows):
        page=pages[row['page_id']]
        assert row['split']==page['split']=='train' and row['label']==0 and row['review_decision']=='unreviewed'
        assert row['source_group']==page['source_group'] and row['parent_image_path']==page['image_path']
        box=row['bbox_xyxy']; x1,y1,x2,y2=box
        assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
        faces=[a['bbox'] for a in page['annotations']]
        if row['label_basis']=='nonmatching_localization_window':
            iou,coverage=geometry(box,faces)
            assert 0<iou<=.3 and coverage<=.6
            assert abs(iou-row['max_face_iou'])<1e-12 and abs(coverage-row['max_face_coverage'])<1e-12
            assert row['parent_positive_id'] in positive_ids
        else:
            assert row['label_basis']=='zero_overlap_expanded_annotations' and not near_face(box,faces)
        assert hashlib.sha256(cache[i,0].tobytes()).hexdigest()==row['crop_sha256']
    ids=np.linspace(0,len(rows)-1,min(24,len(rows)),dtype=int)
    sheet=Image.new('RGB',(1024,366),'white'); draw=ImageDraw.Draw(sheet)
    for cell,index in enumerate(ids):
        x,y=cell%8*128,cell//8*122
        sheet.paste(Image.fromarray(cache[index,0]).resize((96,96)).convert('RGB'),(x+16,y))
        draw.text((x+2,y+98),rows[index]['id'],fill='black')
    sheet_path=args.dataset/'verification_24.png'; sheet.save(sheet_path)
    sizes=np.array([r['bbox_xyxy'][2]-r['bbox_xyxy'][0] for r in rows])
    buckets=np.bincount(np.searchsorted([64,128,256],sizes,side='left'),minlength=4)
    result={'total':len(rows),'works':len({r['source_group'] for r in rows}),'pages':len({r['page_id'] for r in rows}),
            'labels':dict(Counter(r['label_basis'] for r in rows)), 'side_percentiles':np.percentile(sizes,[0,25,50,75,100]).tolist(),
            'size_groups_le64_le128_le256_gt256':buckets.tolist(),
            'candidates_sha256':hashlib.sha256((args.dataset/'candidates.jsonl').read_bytes()).hexdigest(),
            'checks':'all source identities/bounds/train labels/cache hashes; type-specific annotation geometry; localization parent IDs',
            'sheet':str(sheet_path),'sheet_ids':[rows[i]['id'] for i in ids], 'visual_review':'pending; no human approval inferred',
            'command':f'.venv/Scripts/python.exe scripts/verify_auto_window_data.py --dataset {args.dataset} --report {args.report}'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sheet_ids'}))


if __name__=='__main__':
    main()
