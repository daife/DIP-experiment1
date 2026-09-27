"""Refit the saved train verifier after local-context background exclusion."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v2'
    guard=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1'
    pool=ROOT/'datasets/derived/cascade_auto_background_merged_v1'
    output=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v3'
    if output.exists():
        raise ValueError('output exists; no overwrites')
    metadata=json.loads((guard/'provenance.json').read_text())
    summary=json.loads((guard/'summary.json').read_text())
    assert metadata['source_sha256']==sha(pool/'candidates.jsonl')
    assert summary['accepted_indices_sha256']==sha(guard/'accepted_indices.npy')
    assert summary['decisions_sha256']==sha(guard/'decisions.jsonl')
    decisions=list(map(json.loads,(guard/'decisions.jsonl').read_text().splitlines()))
    rows_pool=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
    accepted=np.load(guard/'accepted_indices.npy')
    assert len(rows_pool)==len(decisions)==summary['total']
    assert len(set(accepted))==len(accepted)==summary['accepted']
    assert all(d['array_index']==i and d['id']==rows_pool[i]['id'] for i,d in enumerate(decisions))
    assert set(accepted)=={d['array_index'] for d in decisions if d['decision']=='keep_auto'}
    allowed=set(accepted.tolist());quarantine=set(metadata['quarantined_pages'])
    provenance=json.loads((source/'provenance.json').read_text())
    assert provenance['features_sha256']==sha(source/'features.npz')
    assert provenance['samples_sha256']==sha(source/'samples.jsonl')
    rows=list(map(json.loads,(source/'samples.jsonl').read_text().splitlines()))
    local_rows=list(map(json.loads,(ROOT/'datasets/derived/cascade_localization_negatives_v1/candidates.jsonl').read_text().splitlines()))
    keep=[];rejected=[]
    for i,row in enumerate(rows):
        reason=None
        if row['label']==0 and row.get('dataset')=='cascade_auto_background_merged_v1':
            if row['source_index'] not in allowed:
                reason='background_context_guard'
        elif row['label']==0 and row.get('dataset')=='cascade_localization_negatives_v1':
            if local_rows[row['source_index']]['page_id'] in quarantine:
                reason='confirmed_face_page_negative_quarantine'
        if reason:
            rejected.append({'parent_index':i,'source':row,'reason':reason})
        else:
            keep.append(i)
    with np.load(source/'features.npz') as data:
        x=data['x'][keep];y=data['y'][keep]
    assert int(y.sum())==provenance['positive']
    output.mkdir(parents=True)
    np.save(output/'parent_indices.npy',np.asarray(keep,dtype=np.int64))
    np.savez_compressed(output/'features.npz',x=x,y=y)
    (output/'samples.jsonl').write_text(''.join(json.dumps(rows[i])+'\n' for i in keep))
    (output/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    model=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',cache_size=1024,random_state=20260926)
    model.fit(x,y);joblib.dump(model,output/'model.joblib',compress=3)
    report={'split':'train','seed':20260926,'positive':int(y.sum()),'negative':int((y==0).sum()),'removed_negative':len(rejected),
        'parent_provenance_sha256':sha(source/'provenance.json'),'guard_provenance_sha256':sha(guard/'provenance.json'),
        'guard_summary_sha256':sha(guard/'summary.json'),'features_sha256':sha(output/'features.npz'),
        'samples_sha256':sha(output/'samples.jsonl'),'model_sha256':sha(output/'model.joblib'),'script_sha256':sha(Path(__file__)),
        'C':1,'gamma':'scale','class_weight':'balanced','support_vectors':len(model.support_vectors_),
        'limitation':'Local teacher only checked selected hard backgrounds; other source omissions can remain. Validation/test not used to fit.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
