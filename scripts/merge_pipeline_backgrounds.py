"""Merge accepted old/fresh train backgrounds, retaining both teacher chains."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='cascade_auto_pipeline_backgrounds_merged_v1')
    parser.add_argument('--extra-quarantine',type=Path)
    args=parser.parse_args()
    assert args.tag.replace('_','').isalnum()
    output=ROOT/'datasets/derived'/args.tag
    guard=ROOT/'datasets/derived'/(args.tag+'_guard')
    assert not output.exists() and not guard.exists()
    state=json.loads((ROOT/'results/cascade_auto_pipeline_background_guards.json').read_text())
    assert state['status']=='data_filtering_finished_not_acceptance'
    assert all(s['exit_code']==0 for s in state['stages'])
    review=ROOT/'results/cascade_auto_pipeline_backgrounds_visual_review.json'
    known=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1/provenance.json'
    quarantine=set(json.loads(review.read_text())['quarantined_pages'])
    quarantine.update(json.loads(known.read_text())['quarantined_pages'])
    if args.extra_quarantine:
        quarantine.update(json.loads(args.extra_quarantine.read_text())['quarantined_pages'])
    sources=[('cascade_auto_background_merged_v1','cascade_auto_background_scale_context_guard_v1'),
             ('cascade_auto_pipeline_backgrounds_v1','cascade_auto_pipeline_backgrounds_v1_context_guard')]
    rows=[];channels=[];seen=set();source_reports=[];excluded=[]
    for name,guard_name in sources:
        pool=ROOT/'datasets/derived'/name;source_guard=ROOT/'datasets/derived'/guard_name
        manifest=pool/'candidates.jsonl';array=pool/'channels11.npy'
        meta=json.loads((source_guard/'provenance.json').read_text())
        summary=json.loads((source_guard/'summary.json').read_text())
        assert meta['source_sha256']==sha(manifest)
        assert summary['accepted_indices_sha256']==sha(source_guard/'accepted_indices.npy')
        indices=np.load(source_guard/'accepted_indices.npy')
        source_rows=list(map(json.loads,manifest.read_text().splitlines()))
        source_x=np.load(array,mmap_mode='r')
        assert source_x.shape==(len(source_rows),11,24,24) and source_x.dtype==np.uint8
        assert len(indices)==len(set(indices.tolist())) and np.all((indices>=0)&(indices<len(source_rows)))
        accepted=0
        for i in indices:
            row=source_rows[int(i)]
            assert row['split']=='train' and row['label']==0
            digest=hashlib.sha256(source_x[i,0].tobytes()).hexdigest()
            assert digest==row['crop_sha256']
            if row['page_id'] in quarantine:
                excluded.append({'source_pool':name,'source_index':int(i),'reason':'visual_page_quarantine'});continue
            if digest in seen:
                excluded.append({'source_pool':name,'source_index':int(i),'reason':'duplicate_gray_sha256'});continue
            seen.add(digest)
            rows.append({**row,'id':f'pipeline_merged:{len(rows):05d}','merge_source_pool':name,
                         'merge_source_index':int(i),'merge_source_id':row['id']})
            channels.append(source_x[i].copy());accepted+=1
        source_reports.append({'pool':name,'guard':guard_name,'manifest_sha256':sha(manifest),
                               'channels_sha256':sha(array),'guard_provenance_sha256':sha(source_guard/'provenance.json'),
                               'accepted_indices_sha256':sha(source_guard/'accepted_indices.npy'),
                               'source_accepted':len(indices),'merged_accepted':accepted})
    assert source_reports[1]['merged_accepted']>0
    output.mkdir();guard.mkdir()
    np.save(output/'channels11.npy',np.stack(channels))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    provenance={'sources':source_reports,'negative':len(rows),'quarantined_pages':sorted(quarantine),
                'extra_quarantine_sha256':sha(args.extra_quarantine) if args.extra_quarantine else None,
                'visual_review_sha256':sha(review),'prior_quarantine_sha256':sha(known),
                'excluded':excluded,'script_sha256':sha(Path(__file__)),
                'candidate_sha256':sha(output/'candidates.jsonl'),'channels_sha256':sha(output/'channels11.npy'),
                'limitation':'Composite of preceding teacher acceptance and conservative visual page quarantine; source windows remain unreviewed.'}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    np.save(guard/'accepted_indices.npy',np.arange(len(rows),dtype=np.int64))
    (guard/'provenance.json').write_text(json.dumps({'source_sha256':sha(output/'candidates.jsonl'),
        'composite_source_provenance_sha256':sha(output/'provenance.json'),'quarantined_pages':sorted(quarantine),
        'method':'Only source teacher-accepted indices, then visual page exclusion and global pixel deduplication'},indent=2)+'\n')
    (guard/'summary.json').write_text(json.dumps({'accepted':len(rows),'accepted_indices_sha256':sha(guard/'accepted_indices.npy')},indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':
    main()
