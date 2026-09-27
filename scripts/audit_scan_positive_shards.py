"""Recompute completed positive shards from immutable source pages and GT."""
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
from src.candidate_verifier import hog
from train_candidate_verifier import iou_max


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pool',type=Path,default=ROOT/'datasets/derived/cascade_auto_scan_positives_v1')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('use a new audit output; preserve historical snapshots')
    pool=args.pool.resolve()
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    selection=json.loads((pool/'selection.json').read_text())
    assert selection['manifest_sha256']==sha(manifest)
    pages={p['image_id']:p for p in map(json.loads,manifest.read_text().splitlines())}
    excluded=set()
    for name,digest in selection.get('excluded_selection_shas',{}).items():
        assert sha(ROOT/name)==digest
        excluded.update(json.loads((ROOT/name).read_text())['page_ids'])
    assert not excluded.intersection(selection['page_ids'])
    if selection.get('fresh_pages_per_work'):
        from collections import Counter
        counts_per_work=Counter(pages[i]['source_group'] for i in selection['page_ids'])
        assert len(counts_per_work)==81 and set(counts_per_work.values())=={selection['fresh_pages_per_work']}
    files=sorted(pool.glob('page_*.json'))
    assert files
    checked=[];seen_ids=set();counts={}
    merged_rows=[];merged_channels=[];merged_hog=[];pixels=set()
    for path in files:
        entry=json.loads(path.read_text());page=pages[entry['image_id']]
        number=int(path.stem.split('_')[1])
        assert selection['page_ids'][number]==entry['image_id'] and page['split']=='train'
        image_path=ROOT/'datasets'/page['image_path']
        assert entry['source_image_sha256']==sha(image_path)
        shard=path.with_suffix('.npz')
        assert entry['shard_sha256']==sha(shard)
        gray=cv2.imread(str(image_path),0)
        assert gray is not None and gray.shape==(page['height'],page['width'])
        gt=np.asarray([a['bbox'] for a in page['annotations']])
        with np.load(shard) as data:
            assert data['channels'].shape==(len(entry['rows']),11,24,24)
            assert data['features'].shape==(len(entry['rows']),324)
            assert data['channels'].dtype==np.uint8 and data['features'].dtype==np.float32
            for i,row in enumerate(entry['rows']):
                assert row['id'] not in seen_ids;seen_ids.add(row['id'])
                assert row['page_id']==page['image_id'] and row['split']=='train' and row['label']==1
                assert row['parent_image_path']==page['image_path'] and row['source_group']==page['source_group']
                box=row['bbox_xyxy'];x1,y1,x2,y2=box
                assert all(type(v)==int for v in box)
                assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
                overlap=float(iou_max(np.asarray([box]),gt)[0])
                assert overlap>=.5 and abs(overlap-row['max_annotation_IoU'])<1e-10
                crop=cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                assert hashlib.sha256(crop.tobytes()).hexdigest()==row['crop_sha256']
                assert np.array_equal(np.stack(compute_11_channels(crop)),data['channels'][i])
                assert np.array_equal(hog(crop),data['features'][i])
                assert row['label_basis'] in ('detected_postNMS','annotation_square_jitter')
                counts[row['label_basis']]=counts.get(row['label_basis'],0)+1
                if row['crop_sha256'] not in pixels:
                    pixels.add(row['crop_sha256']);merged_rows.append(row)
                    merged_channels.append(data['channels'][i].copy());merged_hog.append(data['features'][i].copy())
        checked.append({'metadata':str(path.relative_to(ROOT)),'metadata_sha256':sha(path),'shard_sha256':sha(shard),'image_id':page['image_id'],'rows':len(entry['rows'])})
    final_checked=False
    if args.require_complete:
        assert len(files)==len(selection['page_ids'])
        provenance=json.loads((pool/'provenance.json').read_text())
        for key,name in [('candidate_sha256','candidates.jsonl'),('channels_sha256','channels11.npy'),('hog_sha256','hog_features.npy')]:
            assert provenance[key]==sha(pool/name)
        assert provenance['positive']==len(merged_rows)
        assert list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))==merged_rows
        assert np.array_equal(np.load(pool/'channels11.npy'),np.stack(merged_channels))
        assert np.array_equal(np.load(pool/'hog_features.npy'),np.stack(merged_hog))
        final_checked=True
    report={'completed_page_snapshot':len(checked),'samples_checked':len(seen_ids),'counts':counts,
            'all_train_and_GT_IoU50':True,'all_pixels_channels_HOG_recomputed_exactly':True,
            'selection_sha256':sha(pool/'selection.json'),'script_sha256':sha(Path(__file__)),
            'shards':checked,'final_merged_arrays_checked':final_checked,'unique_samples':len(merged_rows),
            'limitation':'Audit of completed shards captured at start; complete pool only if final_merged_arrays_checked true; not visual-label accuracy certification.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='shards'})


if __name__=='__main__':
    main()
