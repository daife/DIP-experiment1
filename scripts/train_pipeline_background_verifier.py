"""Refit v4 with fresh teacher-accepted hard backgrounds and page quarantine."""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.candidate_verifier import hog


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,default=ROOT/'datasets/derived/cascade_auto_pipeline_backgrounds_merged_v1')
    args=parser.parse_args()
    source=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v4'
    pool=args.pool
    output=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v5'
    assert not output.exists()
    parent=json.loads((source/'provenance.json').read_text())
    for key,name in [('features_sha256','features.npz'),('samples_sha256','samples.jsonl'),('model_sha256','model.joblib')]:
        assert parent[key]==sha(source/name)
    meta=json.loads((pool/'provenance.json').read_text())
    assert meta['candidate_sha256']==sha(pool/'candidates.jsonl')
    assert meta['channels_sha256']==sha(pool/'channels11.npy')
    quarantine=set(meta['quarantined_pages'])
    old_rows=list(map(json.loads,(source/'samples.jsonl').read_text().splitlines()))
    with np.load(source/'features.npz') as data:
        old_x=data['x'].copy();old_y=data['y'].copy()
    assert old_x.shape==(len(old_rows),324) and len(old_y)==len(old_rows)
    assert np.array_equal(old_y,np.asarray([r['label'] for r in old_rows]))
    source_rows={}
    keep=[];rejected=[]
    for i,row in enumerate(old_rows):
        if row['label']==0:
            name=row['dataset']
            if name not in source_rows:
                source_rows[name]=list(map(json.loads,(ROOT/'datasets/derived'/name/'candidates.jsonl').read_text().splitlines()))
            candidate=source_rows[name][row['source_index']]
            assert candidate['id']==row['source_id'] and candidate['split']=='train' and candidate['label']==0
            if candidate['page_id'] in quarantine:
                rejected.append({'parent_index':i,'source':row,'reason':'visual_page_quarantine'});continue
        keep.append(i)
    x=old_x[keep];y=old_y[keep];rows=[old_rows[i] for i in keep]
    seen={hashlib.sha256(v.tobytes()).hexdigest() for v in x}
    candidates=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
    channels=np.load(pool/'channels11.npy',mmap_mode='r')
    assert channels.shape==(len(candidates),11,24,24) and channels.dtype==np.uint8
    added=[];extra=[];duplicates=[]
    for i,row in enumerate(candidates):
        # Older accepted backgrounds are preserved through the v4 parent.
        if row['merge_source_pool']!='cascade_auto_pipeline_backgrounds_v1':
            continue
        assert row['split']=='train' and row['label']==0 and row['page_id'] not in quarantine
        assert hashlib.sha256(channels[i,0].tobytes()).hexdigest()==row['crop_sha256']
        vector=hog(channels[i,0])
        digest=hashlib.sha256(vector.tobytes()).hexdigest()
        if digest in seen:
            duplicates.append(i);continue
        seen.add(digest);extra.append(vector);added.append(i)
        rows.append({'dataset':pool.name,'source_index':i,'source_id':row['id'],'label':0,
                     'label_basis':row['label_basis'],'mining_teacher_margin':row['verifier_margin']})
    assert extra,'no fresh unique backgrounds'
    x=np.concatenate([x,np.asarray(extra,dtype=np.float32)])
    y=np.concatenate([y,np.zeros(len(extra),dtype=np.uint8)])
    assert int(y.sum())==parent['positive']
    output.mkdir()
    np.save(output/'parent_indices.npy',np.asarray(keep,dtype=np.int64))
    np.save(output/'added_background_indices.npy',np.asarray(added,dtype=np.int64))
    np.savez_compressed(output/'features.npz',x=x,y=y)
    (output/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (output/'rejected.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rejected))
    model=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',cache_size=1024,random_state=20260926)
    model.fit(x,y);joblib.dump(model,output/'model.joblib',compress=3)
    report={'split':'train','seed':20260926,'positive':int(y.sum()),'negative':int((y==0).sum()),
            'removed_negative':len(rejected),'added_negative':len(added),'duplicate_fresh_indices':duplicates,
            'parent_provenance_sha256':sha(source/'provenance.json'),'pool_provenance_sha256':sha(pool/'provenance.json'),
            'parent_indices_sha256':sha(output/'parent_indices.npy'),'added_indices_sha256':sha(output/'added_background_indices.npy'),
            'features_sha256':sha(output/'features.npz'),'samples_sha256':sha(output/'samples.jsonl'),
            'model_sha256':sha(output/'model.joblib'),'script_sha256':sha(Path(__file__)),
            'C':1,'gamma':'scale','class_weight':'balanced','support_vectors':len(model.support_vectors_),
            'limitation':'Train-only automatic teacher filtering; no validation/test fitting. Accepted windows may still contain undetected unannotated faces.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
