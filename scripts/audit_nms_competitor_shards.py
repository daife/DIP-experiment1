"""Recompute source geometry and pixels for completed NMS competitor shards."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.channels11 import compute_11_channels
from train_box_refiner import ious
from build_localization_negatives import geometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve prior audit snapshots; choose new output')
    pool=ROOT/'datasets/derived/cascade_auto_nms_competitors_v1'
    selection=json.loads((pool/'selection.json').read_text())
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    assert selection['manifest_sha256']==sha(manifest)
    pages={p['image_id']:p for p in map(json.loads,manifest.read_text().splitlines())}
    quarantine_path=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1/provenance.json'
    assert selection['quarantine_sha256']==sha(quarantine_path)
    quarantine=set(json.loads(quarantine_path.read_text())['quarantined_pages'])
    files=sorted(pool.glob('page_*.json'));audited=[];count=0;witness_count=0;ids=set()
    for path in files:
        entry=json.loads(path.read_text());page=pages[entry['image_id']]
        assert selection['page_ids'][int(path.stem.split('_')[1])]==page['image_id']
        assert page['split']=='train' and page['image_id'] not in quarantine
        source=ROOT/'datasets'/page['image_path'];shard=path.with_suffix('.npz')
        assert entry['image_sha256']==sha(source) and entry['shard_sha256']==sha(shard)
        gray=cv2.imread(str(source),0)
        assert gray is not None and gray.shape==(page['height'],page['width'])
        gt=np.asarray([a['bbox'] for a in page['annotations']])
        with np.load(shard) as data:
            channels=data['channels']
            assert channels.shape==(len(entry['rows']),11,24,24) and channels.dtype==np.uint8
            for row,c in zip(entry['rows'],channels,strict=True):
                assert row['id'] not in ids;ids.add(row['id'])
                assert row['split']=='train' and row['page_id']==page['image_id'] and row['label']==0
                assert row['source_group']==page['source_group'] and row['parent_image_path']==page['image_path']
                assert row['label_basis']=='nonmatching_NMS_competitor'
                box=row['bbox_xyxy'];x1,y1,x2,y2=box
                assert all(type(v)==int for v in box)
                assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
                overlap,coverage=geometry(box,gt)
                assert .1<=overlap<.5 and coverage<=.8
                assert abs(overlap-row['max_annotation_IoU'])<1e-10 and abs(coverage-row['max_gt_coverage'])<1e-10
                assert np.isfinite(row['cascade_score']) and row['witnesses']
                for w in row['witnesses']:
                    positive=np.asarray([w['positive_bbox']]);index=w['annotation_index']
                    assert 0<=index<len(gt) and np.all(positive[0,:2]>=0)
                    assert positive[0,2]<=page['width'] and positive[0,3]<=page['height']
                    assert np.all(positive[0,2:]>positive[0,:2])
                    match=float(ious(positive,gt[index:index+1])[0,0]);competition=float(ious(np.asarray([box]),positive)[0,0])
                    assert match>=.5 and competition>.3
                    assert abs(match-w['positive_GT_IoU'])<1e-10 and abs(competition-w['competitor_IoU'])<1e-10
                    assert np.isfinite(w['positive_score']) and row['cascade_score']>=w['positive_score']
                    witness_count+=1
                crop=cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                assert hashlib.sha256(crop.tobytes()).hexdigest()==row['crop_sha256']
                assert np.array_equal(np.stack(compute_11_channels(crop)),c)
                count+=1
        audited.append({'metadata':str(path.relative_to(ROOT)),'metadata_sha256':sha(path),'shard_sha256':sha(shard),'rows':len(entry['rows'])})
    assert files
    report={'completed_page_snapshot':len(files),'samples_checked':count,'witnesses_checked':witness_count,
            'all_train_nonquarantined_and_nonmatching':True,'all_geometry_and_pixels_recomputed':True,
            'selection_sha256':sha(pool/'selection.json'),'script_sha256':sha(Path(__file__)),
            'shards':audited,'limitation':'Saved score ordering checked; not independent replay of full scanning, unique suppression cause, or visual certification.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='shards'})


if __name__=='__main__':
    main()
