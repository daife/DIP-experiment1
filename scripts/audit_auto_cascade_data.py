"""Audit expanded train-only labels and render a reproducible local review sheet."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_detection_dataset import near_face


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/derived/cascade_auto_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'results/cascade_auto_v1_data_audit.json')
    args = parser.parse_args()
    source = ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    pages = {p['image_id']: p for p in map(json.loads, source.read_text(encoding='utf-8').splitlines())}
    rows = list(map(json.loads, (args.dataset/'samples.jsonl').read_text(encoding='utf-8').splitlines()))
    x = np.load(args.dataset/'channels.npy', mmap_mode='r')
    y = np.load(args.dataset/'labels.npy')
    assert x.shape == (len(rows), 11, 24, 24) and x.dtype == np.uint8
    assert len(y) == len(rows)
    assert len({r['id'] for r in rows}) == len(rows)
    groups = {0: Counter(), 1: Counter()}
    counts = {0: Counter(), 1: Counter()}
    for i, row in enumerate(rows):
        page = pages[row['page_id']]
        assert row['split'] == page['split'] == 'train'
        assert row['source_group'] == page['source_group']
        assert row['parent_image_path'] == page['image_path']
        assert row['review_decision'] == 'unreviewed'
        assert y[i] == row['label']
        box = row['bbox_xyxy']
        assert 0 <= box[0] < box[2] <= page['width'] and 0 <= box[1] < box[3] <= page['height']
        if row['label'] == 0:
            assert row['annotation_index'] is None and not row['horizontal_flip']
            assert not near_face(box, [a['bbox'] for a in page['annotations']])
        else:
            assert 0 <= row['annotation_index'] < len(page['annotations'])
        assert hashlib.sha256(x[i, 0].tobytes()).hexdigest() == row['crop_sha256']
        groups[row['label']][row['source_group']] += 1
        counts[row['label']][row['page_id']] += 1
    chosen = []
    for label in (1, 0):
        seen = set()
        for i, row in enumerate(rows):
            if row['label'] == label and row['source_group'] not in seen and len(seen) < 24:
                chosen.append(i)
                seen.add(row['source_group'])
    sheet = Image.new('RGB', (8*128, 6*122), 'white')
    draw = ImageDraw.Draw(sheet)
    for cell, i in enumerate(chosen):
        left, top = cell % 8 * 128, cell // 8 * 122
        sheet.paste(Image.fromarray(x[i, 0]).resize((96, 96)).convert('RGB'), (left+16, top))
        draw.text((left+4, top+98), f"{rows[i]['id']} L{rows[i]['label']}", fill='black')
    sheet_path = args.dataset/'audit_48_samples.png'
    sheet.save(sheet_path)
    report = {'samples': len(rows), 'positive': int(y.sum()), 'negative': int((y==0).sum()),
              'manifest_sha256': hashlib.sha256((args.dataset/'samples.jsonl').read_bytes()).hexdigest(),
              'groups_by_label': {str(k): dict(v) for k,v in groups.items()},
              'pages_by_label': {str(k): len(v) for k,v in counts.items()},
              'max_negative_per_page': max(counts[0].values()),
              'checks': 'all rows: train source identity, bounds, labels, unique IDs, channel0 hash, negative annotation exclusion',
              'sheet': str(sheet_path.relative_to(ROOT)), 'sheet_ids': [rows[i]['id'] for i in chosen],
              'visual_review': 'pending; automated checks cannot rule out unannotated faces'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups_by_label', 'sheet_ids')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
