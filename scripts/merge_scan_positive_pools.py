"""Combine audited positive pools with immutable source-index lineage."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'datasets/derived/cascade_auto_scan_positives_merged_v2'
    assert not output.exists()
    fresh=ROOT/'datasets/derived/cascade_auto_scan_positives_v2'
    audit=ROOT/'results/cascade_auto_scan_positives_v2_final_shard_audit.json'
    checked=json.loads(audit.read_text())
    assert checked['final_merged_arrays_checked'] and checked['completed_page_snapshot']==162
    assert checked['selection_sha256']==sha(fresh/'selection.json')
    assert all(item['metadata_sha256']==sha(ROOT/item['metadata']) and item['shard_sha256']==sha((ROOT/item['metadata']).with_suffix('.npz')) for item in checked['shards'])
    rows=[];channels=[];hogs=[];seen=set();reports=[];duplicates=[]
    for name in ['cascade_auto_scan_positives_v1','cascade_auto_scan_positives_v2']:
        pool=ROOT/'datasets/derived'/name
        meta=json.loads((pool/'provenance.json').read_text())
        for key,file in [('candidate_sha256','candidates.jsonl'),('channels_sha256','channels11.npy'),('hog_sha256','hog_features.npy')]:
            assert meta[key]==sha(pool/file)
        candidates=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
        x=np.load(pool/'channels11.npy',mmap_mode='r');h=np.load(pool/'hog_features.npy',mmap_mode='r')
        assert x.shape==(len(candidates),11,24,24) and x.dtype==np.uint8
        assert h.shape==(len(candidates),324) and h.dtype==np.float32
        added=0
        for i,row in enumerate(candidates):
            assert row['split']=='train' and row['label']==1 and row['max_annotation_IoU']>=.5
            digest=hashlib.sha256(x[i,0].tobytes()).hexdigest()
            assert digest==row['crop_sha256']
            if digest in seen:
                duplicates.append({'pool':name,'source_index':i});continue
            seen.add(digest);added+=1
            rows.append({**row,'id':f'merged_scan_positive:{len(rows):06d}','merge_source_pool':name,
                         'merge_source_index':i,'merge_source_id':row['id']})
            channels.append(x[i].copy());hogs.append(h[i].copy())
        reports.append({'pool':name,'provenance_sha256':sha(pool/'provenance.json'),'source_positive':len(candidates),'added':added})
    assert reports[1]['added']>0
    output.mkdir()
    np.save(output/'channels11.npy',np.stack(channels));np.save(output/'hog_features.npy',np.stack(hogs))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    provenance={'positive':len(rows),'sources':reports,'duplicates':duplicates,'fresh_final_audit_sha256':sha(audit),
                'script_sha256':sha(Path(__file__)),'candidate_sha256':sha(output/'candidates.jsonl'),
                'channels_sha256':sha(output/'channels11.npy'),'hog_sha256':sha(output/'hog_features.npy'),
                'limitation':'GT-matched automatic positives, including repeated-face jitter; no full visual certification.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(provenance,flush=True)


if __name__=='__main__':
    main()
