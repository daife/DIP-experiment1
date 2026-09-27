"""Deterministically broaden the background pool before teacher exclusion."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'datasets/derived/cascade_auto_v1'
    output = ROOT/'datasets/derived/cascade_auto_background_coverage_v1'
    rows = list(map(json.loads, (source/'samples.jsonl').read_text().splitlines()))
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row['label'] == 0:
            assert row['split'] == 'train'
            groups[row['source_group']][row['page_id']].append(index)
    # Eight spread-out pages per work; fixed existing windows, no new labels.
    indices = []
    for work in sorted(groups):
        pages = sorted(groups[work])
        chosen = np.linspace(0, len(pages)-1, min(8, len(pages)), dtype=int)
        for position in chosen:
            indices.extend(groups[work][pages[position]])
    indices = np.asarray(sorted(indices), dtype=np.int64)
    original = np.load(source/'channels.npy', mmap_mode='r')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'source_indices.npy').exists():
        assert np.array_equal(np.load(output/'source_indices.npy'), indices)
    np.save(output/'source_indices.npy', indices)
    with (output/'candidates.jsonl').open('w') as stream:
        for index in indices:
            stream.write(json.dumps(rows[index])+'\n')
    np.save(output/'channels11.npy', original[indices])
    report = {'source_manifest_sha256':hashlib.sha256((source/'samples.jsonl').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection':'eight equally spaced sorted negative-bearing pages per train work; all existing windows on selected pages',
              'samples':len(indices), 'works':len(groups),
              'pages':len({rows[i]['page_id'] for i in indices}),
              'human_reviewed':False, 'random_seed':None}
    (output/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
