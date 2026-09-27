"""Mine high-margin unused windows from already audited train-only pools."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.candidate_verifier import hog


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v1'
    output=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v2'
    output.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((source/'provenance.json').read_text())
    for key,name in [('features_sha256','features.npz'),('samples_sha256','samples.jsonl'),('model_sha256','model.joblib')]:
        assert sha(source/name)==provenance[key]
    old_rows=list(map(json.loads,(source/'samples.jsonl').read_text().splitlines()))
    with np.load(source/'features.npz') as data:
        x,y=data['x'].copy(),data['y'].copy()
    assert len(x)==len(y)==len(old_rows)
    model=joblib.load(source/'model.joblib')
    extra_x=[];extra_rows=[];reports=[]
    for name in ('cascade_auto_background_merged_v1','cascade_localization_negatives_v1'):
        pool=ROOT/'datasets/derived'/name
        rows=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
        cache=np.load(pool/'channels11.npy',mmap_mode='r')
        old_indices={r['source_index'] for r in old_rows if r.get('dataset')==name}
        eligible=np.asarray([i for i in range(len(rows)) if i not in old_indices],dtype=np.int64)
        assert all(rows[i]['split']=='train' and rows[i]['label']==0 for i in eligible)
        vectors=np.stack([hog(cache[i,0]) for i in eligible])
        scores=model.decision_function(vectors)
        order=np.argsort(-scores,kind='stable')[:2000]
        chosen=eligible[order]
        for i,vector,score in zip(chosen,vectors[order],scores[order]):
            row=rows[i]
            assert hashlib.sha256(cache[i,0].tobytes()).hexdigest()==row['crop_sha256']
            extra_x.append(vector)
            extra_rows.append({'dataset':name,'source_index':int(i),'source_id':row['id'],'label':0,
                               'mining_teacher_margin':float(score),'label_basis':row['label_basis']})
        np.savez_compressed(output/(name+'_mining.npz'),eligible_indices=eligible,scores=scores,selected_indices=chosen)
        report={'dataset':name,'manifest_sha256':sha(pool/'candidates.jsonl'),'previous':len(old_indices),
                'unused_scored':len(eligible),'added':len(chosen),'selected_margin_percentiles':np.percentile(scores[order],[0,25,50,75,100]).tolist()}
        reports.append(report);print(report,flush=True)
    x=np.concatenate([x,np.asarray(extra_x,dtype=np.float32)])
    y=np.concatenate([y,np.zeros(len(extra_x),dtype=np.uint8)])
    np.savez_compressed(output/'features.npz',x=x,y=y)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old_rows+extra_rows))
    seed=20260926
    new=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',cache_size=1024,random_state=seed)
    new.fit(x,y);joblib.dump(new,output/'model.joblib',compress=3)
    report={'split':'train','seed':seed,'positive':int(y.sum()),'negative':int((y==0).sum()),
        'mining':'top 2000 unused margins per audited pool, stable source-index tie order; existing reservoir mining, not fresh page scanning',
        'pools':reports,'parent_provenance_sha256':sha(source/'provenance.json'),'parent_model_sha256':sha(source/'model.joblib'),
        'features_sha256':sha(output/'features.npz'),'samples_sha256':sha(output/'samples.jsonl'),
        'model_sha256':sha(output/'model.joblib'),'script_sha256':sha(Path(__file__)),
        'C':1,'gamma':'scale','class_weight':'balanced','support_vectors':len(new.support_vectors_),
        'limitation':'Inherited automatically labelled pools and train proposal positives; source annotation omissions and teacher errors remain possible. No validation/test used for selection or fitting.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
