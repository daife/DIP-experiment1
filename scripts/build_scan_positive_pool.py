"""Build train-only positives from native Cascade proposals and GT square jitter."""
import hashlib
import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.channels11 import compute_11_channels
from src.candidate_verifier import hog
from src.multiscale import detect_multiscale, PyramidConfig
from train_candidate_verifier import choose_pages, iou_max


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/derived/cascade_auto_scan_positives_v1')
    parser.add_argument('--cascade',type=Path,default=ROOT/'datasets/derived/cascade_auto_scale_context_localization_v1/model.json')
    parser.add_argument('--fresh-pages-per-work',type=int,default=0)
    args=parser.parse_args()
    assert args.fresh_pages_per_work>=0
    output = args.output
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    cascade_path = args.cascade
    exclusions={}
    if args.fresh_pages_per_work:
        excluded=set();groups=defaultdict(list)
        for name in ['cascade_auto_scan_positives_v1','cascade_auto_pipeline_backgrounds_v1']:
            path=ROOT/'datasets/derived'/name/'selection.json'
            exclusions[str(path.relative_to(ROOT))]=sha(path)
            excluded.update(json.loads(path.read_text())['page_ids'])
        for page in map(json.loads,manifest.read_text().splitlines()):
            if page['split']=='train' and page['annotations'] and page['image_id'] not in excluded:
                groups[page['source_group']].append(page)
        assert len(groups)==81
        pages=[]
        for name,group in sorted(groups.items()):
            ranked=sorted(group,key=lambda p:hashlib.sha256(('scan-positive-v2:'+p['image_id']).encode()).digest())
            assert len(ranked)>=args.fresh_pages_per_work
            pages.extend(ranked[:args.fresh_pages_per_work])
        assert len(pages)==81*args.fresh_pages_per_work
    else:
        pages = choose_pages(manifest, 81)
        assert len(pages) == len({p['source_group'] for p in pages}) == 81
    spec = {'manifest_sha256':sha(manifest), 'cascade_sha256':sha(cascade_path),
            'script_sha256':sha(Path(__file__)), 'seed':20260926, 'page_ids':[p['image_id'] for p in pages],
            'scale_factor':1.2, 'step':2, 'nms_iou':.3, 'minimum_positive_iou':.5,
            'jitter_per_gt':4, 'jitter_scales':[.85,1,1.15,1.3], 'jitter_offsets':[-.1,0,.1],
            'crop_geometry':'native mapped box resized INTER_AREA to 24; same geometry as HOG verification, not identical to pyramid feature patch',
            'limitation':'GT-matched automatic positives; no claim of all-image human review. Jitter positives are not detected proposals.'}
    if args.fresh_pages_per_work:
        spec.update(fresh_pages_per_work=args.fresh_pages_per_work,excluded_selection_shas=exclusions,
                    page_selection='train only; first scan-positive-v2 SHA pages per work, excluding prior positive and pipeline background pages')
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output/'selection.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == spec
    else:
        spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    cascade = cascade_from_dict(json.loads(cascade_path.read_text()))
    progress = []
    for number, page in enumerate(pages):
        assert page['split'] == 'train'
        shard = output/f'page_{number:03d}.npz'
        metadata = output/f'page_{number:03d}.json'
        image_path = ROOT/'datasets'/page['image_path']
        if shard.exists() and metadata.exists():
            entry = json.loads(metadata.read_text())
            assert entry['image_id'] == page['image_id'] and entry['source_image_sha256'] == sha(image_path)
            assert entry['shard_sha256'] == sha(shard)
        else:
            gray = cv2.imread(str(image_path), 0)
            if gray is None:
                raise ValueError(image_path)
            assert gray.shape == (page['height'],page['width'])
            gt = np.asarray([a['bbox'] for a in page['annotations']]).reshape(-1,4)
            boxes, scores, _ = detect_multiscale(gray, cascade, PyramidConfig(1.2,2,.3))
            overlaps = iou_max(boxes, gt)
            selected = [(box.tolist(),'detected_postNMS',float(score)) for box,score,iou in zip(boxes,scores,overlaps) if iou >= .5]
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(('scan-positive-v1:'+page['image_id']).encode()).digest()[:8],'little'))
            for face in gt:
                cx,cy = (face[:2]+face[2:])/2
                side = max(face[2:]-face[:2])
                variants = []
                for scale in spec['jitter_scales']:
                    for dx in spec['jitter_offsets']:
                        for dy in spec['jitter_offsets']:
                            size = max(24,round(side*scale))
                            x,y = round(cx+dx*side-size/2),round(cy+dy*side-size/2)
                            box = [x,y,x+size,y+size]
                            if x>=0 and y>=0 and x+size<=gray.shape[1] and y+size<=gray.shape[0]:
                                if iou_max(np.asarray([box]), np.asarray([face]))[0] >= .5:
                                    variants.append(box)
                if variants:
                    for index in rng.permutation(len(variants))[:4]:
                        selected.append((variants[index],'annotation_square_jitter',None))
            channels, vectors, rows, seen = [], [], [], set()
            for box,basis,score in selected:
                if tuple(box) in seen:
                    continue
                seen.add(tuple(box))
                x1,y1,x2,y2 = box
                crop = cv2.resize(gray[y1:y2,x1:x2],(24,24),interpolation=cv2.INTER_AREA)
                match = float(iou_max(np.asarray([box]),gt)[0])
                assert match >= .5
                channels.append(np.stack(compute_11_channels(crop)))
                vectors.append(hog(crop))
                rows.append({'id':f'scan_positive:{number:03d}:{len(rows):05d}', 'page_id':page['image_id'],
                    'parent_image_path':page['image_path'], 'source_group':page['source_group'], 'split':'train',
                    'bbox_xyxy':box, 'label':1, 'label_basis':basis, 'max_annotation_IoU':match,
                    'cascade_score':score, 'crop_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                    'review_decision':'unreviewed'})
            np.savez_compressed(shard, channels=np.asarray(channels,dtype=np.uint8).reshape(-1,11,24,24),
                                features=np.asarray(vectors,dtype=np.float32).reshape(-1,324))
            entry = {'image_id':page['image_id'], 'source_image_sha256':sha(image_path), 'shard_sha256':sha(shard),
                     'GT':len(gt), 'proposals':len(boxes), 'rows':rows}
            metadata.write_text(json.dumps(entry,indent=2)+'\n')
        progress.append({k:v for k,v in entry.items() if k!='rows'} | {'positives':len(entry['rows'])})
        (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(progress[-1],flush=True)
    rows, channels, features, seen = [], [], [], set()
    for number in range(len(pages)):
        entry = json.loads((output/f'page_{number:03d}.json').read_text())
        with np.load(output/f'page_{number:03d}.npz') as data:
            assert len(entry['rows']) == len(data['channels']) == len(data['features'])
            for row,c,f in zip(entry['rows'],data['channels'],data['features']):
                assert hashlib.sha256(c[0].tobytes()).hexdigest() == row['crop_sha256']
                if row['crop_sha256'] in seen:
                    continue
                seen.add(row['crop_sha256']); rows.append(row); channels.append(c.copy()); features.append(f.copy())
    np.save(output/'channels11.npy',np.stack(channels))
    np.save(output/'hog_features.npy',np.stack(features))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = {**spec, 'positive':len(rows), 'detected_positive':sum(r['label_basis']=='detected_postNMS' for r in rows),
              'jitter_positive':sum(r['label_basis']=='annotation_square_jitter' for r in rows),
              'pages':progress, 'candidate_sha256':sha(output/'candidates.jsonl'),
              'channels_sha256':sha(output/'channels11.npy'), 'hog_sha256':sha(output/'hog_features.npy')}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print({'positive':len(rows),'detected_positive':report['detected_positive'],'jitter_positive':report['jitter_positive']},flush=True)


if __name__ == '__main__':
    main()
