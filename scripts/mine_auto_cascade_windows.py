"""Mine fresh multiscale train-page false positives without fabricated review."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cascade import cascade_from_dict
from src.channels11 import compute_11_channels
from src.multiscale import detect_multiscale, PyramidConfig
from prepare_detection_dataset import near_face


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pages', type=int, default=128)
    parser.add_argument('--per-page', type=int, default=80)
    parser.add_argument('--max-dimension', type=int, default=768)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--scale-strata', action='store_true', help='Interleave ranked native box sizes <=64, <=128, <=256, >256')
    args = parser.parse_args()
    if min(args.pages, args.per_page, args.max_dimension) < 1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'candidates.jsonl').exists():
        parser.error('output already contains candidates; use a new directory')
    manifest = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    groups = defaultdict(list)
    for page in map(json.loads, manifest.read_text(encoding='utf-8').splitlines()):
        if page['split'] == 'train':
            groups[page['source_group']].append(page)
    key = lambda p: hashlib.sha256(f"auto-mining:{args.seed}:{p['image_id']}".encode()).digest()
    for group in groups.values():
        group.sort(key=key, reverse=True)
    selected = []
    while len(selected) < args.pages:
        added = False
        for name in sorted(groups):
            if groups[name] and len(selected) < args.pages:
                selected.append(groups[name].pop())
                added = True
        if not added:
            break
    model = cascade_from_dict(json.loads(args.model.read_text(encoding='utf-8')))
    provenance = {'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'seed': args.seed, 'step': 2, 'scale_factor': 1.2, 'nms_iou': .3,
                  'max_dimension': args.max_dimension, 'per_page': args.per_page,
                  'scale_strata': args.scale_strata, 'native_scale_boundaries': [64,128,256] if args.scale_strata else None,
                  'selected_page_ids': [p['image_id'] for p in selected],
                  'human_reviewed': False, 'label_limitation': 'Source annotations can omit faces.'}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    rows, channels, traces, seen = [], [], [], set()
    audit = {'face_overlap': 0, 'low_texture': 0, 'duplicate': 0}
    for page in selected:
        gray = cv2.imread(str(ROOT/'datasets'/page['image_path']), 0)
        if gray is None:
            raise ValueError(page['image_path'])
        ratio = min(1., args.max_dimension/max(gray.shape))
        scaled = cv2.resize(gray, (round(gray.shape[1]*ratio), round(gray.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
        boxes, scores, layers = detect_multiscale(scaled, model, PyramidConfig(step=2))
        faces = [a['bbox'] for a in page['annotations']]
        accepted = 0
        ranked = np.argsort(-scores, kind='stable')
        if args.scale_strata:
            native_sides = (boxes[:,2]-boxes[:,0])*gray.shape[1]/scaled.shape[1]
            bucket_ids = np.searchsorted([64,128,256],native_sides,side='left')
            buckets = [ranked[bucket_ids[ranked]==bucket] for bucket in range(4)]
            ranked = [bucket[i] for i in range(max(map(len,buckets),default=0)) for bucket in buckets if i<len(bucket)]
        accepted_scale = [0,0,0,0]
        for index in ranked:
            box = np.rint(boxes[index]*np.array([gray.shape[1]/scaled.shape[1], gray.shape[0]/scaled.shape[0]]*2)).astype(int).tolist()
            if near_face(box, faces):
                audit['face_overlap'] += 1
                continue
            x1,y1,x2,y2 = box
            crop = cv2.resize(gray[y1:y2,x1:x2], (24,24), interpolation=cv2.INTER_AREA)
            if crop.std() < 24 or not .04 <= np.mean(crop<220) <= .85:
                audit['low_texture'] += 1
                continue
            digest = hashlib.sha256(crop.tobytes()).hexdigest()
            if digest in seen:
                audit['duplicate'] += 1
                continue
            seen.add(digest)
            row = {'id': f'window{len(rows):07d}', 'page_id': page['image_id'],
                   'parent_image_path': page['image_path'], 'source_group': page['source_group'],
                   'bbox_xyxy': box, 'score': float(scores[index]), 'split': 'train', 'label': 0,
                   'review_decision': 'unreviewed', 'label_basis': 'zero_overlap_expanded_annotations',
                   'crop_sha256': digest}
            rows.append(row)
            channels.append(np.stack(compute_11_channels(crop)))
            accepted_scale[int(np.searchsorted([64,128,256],box[2]-box[0],side='left'))] += 1
            accepted += 1
            if accepted == args.per_page:
                break
        traces.append({'page_id': page['image_id'], 'scaled_size': list(scaled.shape[::-1]),
                       'layers': layers, 'candidates_after_nms': len(boxes), 'accepted': accepted,
                       'accepted_native_scale_counts':accepted_scale})
        print(f"{page['image_id']}: {len(boxes)} candidates, {accepted} mined, total {len(rows)}", flush=True)
        (args.output/'progress.json').write_text(json.dumps({'pages_done':len(traces),'candidates':len(rows),'audit':audit})+'\n')
    if not rows:
        raise ValueError('no eligible mined windows')
    np.save(args.output/'channels11.npy', np.stack(channels))
    with (args.output/'candidates.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    (args.output/'page_traces.json').write_text(json.dumps(traces, indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps({**audit,'total':len(rows),'pages':len(selected),'fully_reviewed':False}, indent=2)+'\n')
    sheet = Image.new('RGB',(8*128,3*122),'white')
    draw = ImageDraw.Draw(sheet)
    for cell,i in enumerate(np.linspace(0,len(rows)-1,min(24,len(rows)),dtype=int)):
        left,top = cell%8*128,cell//8*122
        sheet.paste(Image.fromarray(channels[i][0]).resize((96,96)).convert('RGB'),(left+16,top))
        draw.text((left+2,top+98),rows[i]['id'],fill='black')
    sheet.save(args.output/'contact_sheet.png')


if __name__ == '__main__':
    main()
