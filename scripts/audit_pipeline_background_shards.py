"""Audit completed fresh-background shards against original train pages."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.channels11 import compute_11_channels
from src.candidate_verifier import features
from prepare_detection_dataset import near_face


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contact', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    pool = ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_v1'
    selection = json.loads((pool/'selection.json').read_text())
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    assert sha(manifest) == selection['manifest_sha256']
    pages = {p['image_id']:p for p in map(json.loads, manifest.read_text().splitlines())}
    excluded = set(selection['quarantined_pages'])
    prior = ROOT/'datasets/derived/cascade_auto_scan_positives_v1/selection.json'
    assert sha(prior) == selection['prior_selection_sha256']
    excluded.update(json.loads(prior.read_text())['page_ids'])
    for path, digest in selection['quarantine_shas'].items():
        assert sha(ROOT/path) == digest
    count = 0
    snapshots = []
    contact_rows = []
    tiles = []
    ids = set()
    merged_rows=[];merged_channels=[];merged_features=[];pixel_seen=set()
    for path in sorted(pool.glob('page_*.json')):
        entry = json.loads(path.read_text())
        page = pages[entry['image_id']]
        assert page['split'] == 'train' and page['image_id'] not in excluded
        assert selection['page_ids'][int(path.stem.split('_')[1])] == page['image_id']
        source = ROOT/'datasets'/page['image_path']
        assert sha(source) == entry['image_sha256']
        shard = path.with_suffix('.npz')
        assert sha(shard) == entry['shard_sha256']
        gray = cv2.imread(str(source), 0)
        assert gray is not None and gray.shape == (page['height'], page['width'])
        rows = entry['rows']
        assert len(rows) <= 12
        gt = [a['bbox'] for a in page['annotations']]
        with np.load(shard) as data:
            assert data['channels'].shape == (len(rows),11,24,24)
            assert data['channels'].dtype == np.uint8
            assert data['features'].shape == (len(rows),324)
            assert data['features'].dtype == np.float32
            previous = float('inf')
            for row, channels, hog in zip(rows, data['channels'], data['features'], strict=True):
                assert row['id'] not in ids
                ids.add(row['id'])
                assert row['split']=='train' and row['label']==0 and row['review_decision']=='unreviewed'
                assert row['label_basis']=='zero_overlap_expanded_annotations'
                assert row['page_id']==page['image_id'] and row['source_group']==page['source_group']
                assert row['parent_image_path']==page['image_path']
                box = row['bbox_xyxy']
                x1,y1,x2,y2 = box
                assert all(type(v)==int for v in box)
                assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
                assert not near_face(box,gt)
                margin = row['verifier_margin']
                assert np.isfinite(margin) and .5<=margin<=previous
                previous = margin
                assert np.isfinite(row['cascade_score'])
                crop = cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                assert hashlib.sha256(crop.tobytes()).hexdigest()==row['crop_sha256']
                assert np.array_equal(np.stack(compute_11_channels(crop)),channels)
                assert np.array_equal(features(gray,np.asarray([box]))[0],hog)
                count += 1
                if row['crop_sha256'] not in pixel_seen:
                    pixel_seen.add(row['crop_sha256'])
                    merged_rows.append(row);merged_channels.append(channels.copy());merged_features.append(hog.copy())
                if args.contact and len(tiles)<24 and len(tiles)<2*(len(snapshots)+1):
                    tile = np.full((144,120),255,np.uint8)
                    tile[:120] = cv2.resize(gray[y1:y2,x1:x2],(120,120),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile,f'{len(tiles):02d} m={margin:.2f}',(3,137),cv2.FONT_HERSHEY_SIMPLEX,.4,0,1)
                    tiles.append(tile)
                    contact_rows.append(row)
        snapshots.append({'metadata':str(path.relative_to(ROOT)),'metadata_sha256':sha(path),'shard_sha256':sha(shard),'rows':len(rows)})
    assert snapshots
    final_checked=False
    if args.require_complete:
        assert len(snapshots)==len(selection['page_ids'])==81
        provenance=json.loads((pool/'provenance.json').read_text())
        for key,name in [('candidate_sha256','candidates.jsonl'),('channels_sha256','channels11.npy'),('hog_sha256','hog_features.npy')]:
            assert provenance[key]==sha(pool/name)
        assert provenance['negative']==len(merged_rows)
        assert list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))==merged_rows
        assert np.array_equal(np.load(pool/'channels11.npy'),np.stack(merged_channels))
        assert np.array_equal(np.load(pool/'hog_features.npy'),np.stack(merged_features))
        final_checked=True
    report = {'completed_page_snapshot':len(snapshots),'samples_checked':count,
              'selection_sha256':sha(pool/'selection.json'),'script_sha256':sha(Path(__file__)),
              'geometry_channels_hog_recomputed':True,'shards':snapshots,
              'contact_rows':contact_rows,'final_merged_arrays_checked':final_checked,
              'unique_samples':len(merged_rows),'limitation':'No independent replay of Cascade/HOG scores or visual certification; complete pool only if final_merged_arrays_checked is true.'}
    if args.contact:
        assert len(tiles)==24 and not args.contact.exists()
        canvas = np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,24,6)])
        assert cv2.imwrite(str(args.contact),canvas)
        report.update(contact_path=str(args.contact),contact_sha256=sha(args.contact))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print({'pages':len(snapshots),'samples':count,'contact_samples':len(tiles)})


if __name__=='__main__':
    main()
