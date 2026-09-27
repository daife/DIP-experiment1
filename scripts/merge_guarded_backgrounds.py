"""Merge teacher-vetoed train background pools with immutable provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_detection_dataset import near_face

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = ['cascade_auto_windows_scale_v2', 'cascade_auto_background_coverage_v1']
    output = ROOT/'datasets/derived/cascade_auto_background_merged_v1'
    if output.exists():
        raise ValueError('use a new output; existing merge is immutable')
    pages = {r['image_id']:r for r in map(json.loads, (ROOT/'datasets/manifests/normalized/manga109_faces.jsonl').read_text(encoding='utf-8').splitlines())}
    combined, crops, provenance, seen = [], [], [], set()
    for name in sources:
        source = ROOT/'datasets/derived'/name
        guard = ROOT/'datasets/derived'/(name+'_guard')
        metadata = json.loads((guard/'provenance.json').read_text())
        summary = json.loads((guard/'summary.json').read_text())
        assert metadata['source_sha256'] == sha(source/'candidates.jsonl')
        assert summary['accepted_indices_sha256'] == sha(guard/'accepted_indices.npy')
        assert summary['decisions_sha256'] == sha(guard/'decisions.jsonl')
        rows = list(map(json.loads, (source/'candidates.jsonl').read_text().splitlines()))
        decisions = list(map(json.loads, (guard/'decisions.jsonl').read_text().splitlines()))
        indices = np.load(guard/'accepted_indices.npy')
        assert len(decisions) == summary['total'] == len(rows)
        assert len(indices) == summary['accepted'] and len(set(indices)) == len(indices)
        assert np.all((indices >= 0) & (indices < len(rows)))
        assert set(indices.tolist()) == {r['array_index'] for r in decisions if r['decision'] == 'keep_auto'}
        assert all(d['array_index']==i and d['id']==rows[i]['id'] for i,d in enumerate(decisions))
        data = np.load(source/'channels11.npy', mmap_mode='r')
        assert data.shape == (len(rows),11,24,24) and data.dtype == np.uint8
        count = 0
        for index in indices:
            row = rows[index]
            page = pages[row['page_id']]
            assert row['split']==page['split']=='train' and row['label']==0
            assert row['label_basis']=='zero_overlap_expanded_annotations'
            assert row['parent_image_path']==page['image_path'] and row['source_group']==page['source_group']
            x1,y1,x2,y2 = row['bbox_xyxy']
            assert 0<=x1<x2<=page['width'] and 0<=y1<y2<=page['height']
            assert not near_face(row['bbox_xyxy'], [a['bbox'] for a in page['annotations']])
            digest = hashlib.sha256(data[index,0].tobytes()).hexdigest()
            assert digest == row['crop_sha256']
            if digest in seen:
                continue
            seen.add(digest)
            combined.append({**row, 'id':name+':'+row['id'], 'source_candidate_id':row['id'],
                             'source_dataset':name, 'source_array_index':int(index)})
            crops.append(data[index].copy())
            count += 1
        provenance.append({'dataset':name, 'retained_after_dedup':count,
                           'candidate_sha256':sha(source/'candidates.jsonl'),
                           'guard_provenance_sha256':sha(guard/'provenance.json'),
                           'guard_summary_sha256':sha(guard/'summary.json'),
                           'indices_sha256':sha(guard/'accepted_indices.npy')})
    output.mkdir(parents=True)
    with (output/'candidates.jsonl').open('w') as stream:
        for row in combined:
            stream.write(json.dumps(row)+'\n')
    np.save(output/'channels11.npy', np.stack(crops))
    report = {'sources':provenance, 'total':len(combined),
              'pages':len({r['page_id'] for r in combined}), 'works':len({r['source_group'] for r in combined}),
              'script_sha256':sha(Path(__file__)), 'human_reviewed':False,
              'limitation':'Teacher-vetoed backgrounds; residual unlabeled faces remain possible.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
