"""Await a live train-positive collector, then refit and compare Cascade/HOG."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC
from run_auto_coverage_iteration import wait_for_process

ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT/'datasets/derived/cascade_auto_scan_positives_v1'
VERIFIER=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_verifier(parent=None):
    parent=parent or ROOT/'datasets/derived/cascade_auto_proposal_verifier_v3'
    if VERIFIER.exists():
        raise ValueError('immutable verifier output exists')
    meta=json.loads((parent/'provenance.json').read_text())
    assert meta['features_sha256']==sha(parent/'features.npz')
    assert meta['samples_sha256']==sha(parent/'samples.jsonl')
    pool_meta=json.loads((POOL/'provenance.json').read_text())
    assert pool_meta['candidate_sha256']==sha(POOL/'candidates.jsonl')
    assert pool_meta['hog_sha256']==sha(POOL/'hog_features.npy')
    rows=[json.loads(line) for line in (parent/'samples.jsonl').read_text().splitlines()]
    additional=[json.loads(line) for line in (POOL/'candidates.jsonl').read_text().splitlines()]
    assert all(r['split']=='train' and r['label']==1 and r['max_annotation_IoU']>=.5 for r in additional)
    with np.load(parent/'features.npz') as data:
        x,y=data['x'].copy(),data['y'].copy()
    assert len(rows)==len(x)==len(y)
    new=np.load(POOL/'hog_features.npy')
    assert new.shape==(len(additional),324)
    seen={hashlib.sha256(vector.tobytes()).digest() for vector in x}
    indices=[]
    for i,vector in enumerate(new):
        digest=hashlib.sha256(vector.tobytes()).digest()
        if digest not in seen:
            indices.append(i);seen.add(digest)
    assert indices
    x=np.concatenate([x,new[indices]]);y=np.concatenate([y,np.ones(len(indices),dtype=np.uint8)])
    rows.extend({'dataset':POOL.name,'source_index':i,'source_id':additional[i]['id'],'label':1,'split':'train'} for i in indices)
    VERIFIER.mkdir(parents=True)
    np.save(VERIFIER/'added_positive_indices.npy',np.asarray(indices,dtype=np.int64))
    np.savez_compressed(VERIFIER/'features.npz',x=x,y=y)
    (VERIFIER/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    model=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',cache_size=1024,random_state=20260926)
    model.fit(x,y);joblib.dump(model,VERIFIER/'model.joblib',compress=3)
    report={'seed':20260926,'positive':int(y.sum()),'negative':int((y==0).sum()),'added_positive':len(indices),
        'parent_provenance_sha256':sha(parent/'provenance.json'),'pool_provenance_sha256':sha(POOL/'provenance.json'),
        'features_sha256':sha(VERIFIER/'features.npz'),'samples_sha256':sha(VERIFIER/'samples.jsonl'),
        'model_sha256':sha(VERIFIER/'model.joblib'),'script_sha256':sha(Path(__file__)),
        'support_vectors':len(model.support_vectors_),'C':1,'gamma':'scale','class_weight':'balanced',
        'limitation':'Automatic train labels; existing negative omissions may remain; no validation/test fitting.'}
    (VERIFIER/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


def main():
    global POOL,VERIFIER
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--fit-verifier',action='store_true')
    parser.add_argument('--pool',type=Path)
    parser.add_argument('--verifier-output',type=Path)
    parser.add_argument('--parent-verifier',type=Path)
    args=parser.parse_args()
    if args.fit_verifier:
        POOL=args.pool or POOL
        VERIFIER=args.verifier_output or VERIFIER
        fit_verifier(args.parent_verifier);return
    if args.pool or args.verifier_output or args.parent_verifier:
        parser.error('custom verifier paths require --fit-verifier')
    if not args.wait_pid:
        parser.error('--wait-pid is required for iteration')
    record=ROOT/'results/cascade_auto_scan_positive_iteration.json'
    if record.exists():
        raise ValueError('immutable run record exists')
    report={'status':'waiting_for_positive_collection','collector_pid':args.wait_pid,'stages':[]}
    def save():
        record.write_text(json.dumps(report,indent=2)+'\n')
    save()
    output=ROOT/'datasets/derived/cascade_auto_scan_positive_localization_v1'
    guard=ROOT/'datasets/derived/cascade_auto_background_scale_context_guard_v1'
    quarantine=ROOT/'datasets/derived/cascade_auto_background_context_guard_v1/provenance.json'
    ordinary='results/cascade_auto_scan_positive_localization_v1_validation20.json'
    def combined(cascade,reference,verifier,name,prediction):
        return ['scripts/evaluate_combined_improvements.py','--pages','20','--pages-per-work','2','--cascade',str(cascade),'--reference',str(reference),'--tight-verifier',str(verifier),'--extra-thresholds','0','.25','.5','.75','1','1.25','2','--output',f'results/{name}.json','--prediction-dir',str(prediction)]
    try:
        wait_for_process(args.wait_pid)
        assert (POOL/'provenance.json').exists(), 'collector did not finish all shards'
        commands=[
            ['scripts/run_scan_positive_iteration.py','--fit-verifier'],
            ['scripts/train_auto_mined_cascade.py','--mined','datasets/derived/cascade_auto_background_merged_v1','--mined-guard',str(guard),'--negative-quarantine',str(quarantine),'--localization-negatives','datasets/derived/cascade_localization_negatives_v1','--proposal-positives',str(POOL),'--base-negative-count','0','--positive-hog-min','0','--output',str(output),'--stages','3','--trees','64','--root-candidates','64','--child-candidates','32','--stage-recall','.98','--histogram-splits'],
            ['scripts/evaluate_candidate_verifier.py','--pages','20','--pages-per-work','2','--cascade',str(output/'model.json'),'--verifier','models/candidate_hog_rbf_step3_v1.joblib','--output',ordinary,'--cache-dir',str(output/'validation_cache')],
            combined('datasets/derived/cascade_auto_scale_context_localization_v1/model.json','results/cascade_auto_scale_context_localization_v1_validation20.json',VERIFIER/'model.joblib','cascade_auto_scan_positive_verifier_v4_validation20',VERIFIER/'validation20'),
            combined(output/'model.json',ordinary,'datasets/derived/cascade_auto_proposal_verifier_v3/model.joblib','cascade_auto_scan_positive_localization_v1_v3_combined20',output/'combined20_v3'),
            combined(output/'model.json',ordinary,VERIFIER/'model.joblib','cascade_auto_scan_positive_localization_v1_v4_combined20',output/'combined20_v4'),
        ]
        for i,command in enumerate(commands):
            entry={'command':[sys.executable,'-u',*command],'status':'running','stdout':f'tmp/cascade_auto_scan_positive_stage{i}.stdout.log','stderr':f'tmp/cascade_auto_scan_positive_stage{i}.stderr.log'}
            report['status']='running';report['stages'].append(entry);save()
            with (ROOT/entry['stdout']).open('w') as stdout,(ROOT/entry['stderr']).open('w') as stderr:
                result=subprocess.run(entry['command'],cwd=ROOT,stdout=stdout,stderr=stderr)
            entry.update(exit_code=result.returncode,status='completed' if result.returncode==0 else 'failed');save()
            if result.returncode:
                raise RuntimeError(f'stage {i} failed')
        report['status']='experiments_finished_not_acceptance';save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
