"""Refit immutable HOG training rows with an explicit SVC C control."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--C',type=float,required=True)
    args=p.parse_args()
    assert np.isfinite(args.C) and args.C>0 and not args.output.exists()
    meta=json.loads((args.source/'provenance.json').read_text())
    for key,name in [('features_sha256','features.npz'),('samples_sha256','samples.jsonl'),('model_sha256','model.joblib')]:
        assert meta[key]==sha(args.source/name)
    rows=list(map(json.loads,(args.source/'samples.jsonl').read_text().splitlines()))
    with np.load(args.source/'features.npz') as data:
        x=data['x'].copy();y=data['y'].copy()
    assert x.shape==(len(rows),324) and len(y)==len(rows) and np.isfinite(x).all()
    assert np.array_equal(y,np.asarray([r['label'] for r in rows]))
    assert int(y.sum())==meta['positive'] and int((y==0).sum())==meta['negative']
    args.output.mkdir()
    for name in ['features.npz','samples.jsonl']:
        shutil.copy2(args.source/name,args.output/name)
        assert sha(args.source/name)==sha(args.output/name)
    model=SVC(kernel='rbf',C=args.C,gamma='scale',class_weight='balanced',cache_size=1024,random_state=20260926)
    model.fit(x,y);joblib.dump(model,args.output/'model.joblib',compress=3)
    report={'split':'train','seed':20260926,'positive':int(y.sum()),'negative':int((y==0).sum()),
            'C':args.C,'gamma':'scale','class_weight':'balanced','support_vectors':len(model.support_vectors_),
            'parent_provenance_sha256':sha(args.source/'provenance.json'),'parent_model_sha256':sha(args.source/'model.joblib'),
            'features_sha256':sha(args.output/'features.npz'),'samples_sha256':sha(args.output/'samples.jsonl'),
            'model_sha256':sha(args.output/'model.joblib'),'script_sha256':sha(Path(__file__)),
            'limitation':'Capacity control on exact inherited training rows, not a new data-mining iteration or blind validation.'}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
