"""Fit box refinement from audited train-only scan/jitter positives."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_box_refiner import ious


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pool=ROOT/'datasets/derived/cascade_auto_scan_positives_merged_v2'
    output=ROOT/'datasets/derived/cascade_auto_box_refiner_expanded_v1'
    assert not output.exists()
    meta=json.loads((pool/'provenance.json').read_text())
    audit=ROOT/'results/cascade_auto_scan_positives_merged_v2_audit.json'
    checked=json.loads(audit.read_text())
    assert checked['provenance_sha256']==sha(pool/'provenance.json')
    assert checked['source_rows_channels_hog_exact'] and checked['all_train_positive_GT_IoU50']
    assert meta['candidate_sha256']==sha(pool/'candidates.jsonl') and meta['hog_sha256']==sha(pool/'hog_features.npy')
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    pages={p['image_id']:p for p in map(json.loads,manifest.read_text().splitlines())}
    rows=list(map(json.loads,(pool/'candidates.jsonl').read_text().splitlines()))
    x=np.load(pool/'hog_features.npy')
    assert x.shape==(len(rows),324) and x.dtype==np.float32
    targets=[];provenance=[]
    for i,row in enumerate(rows):
        page=pages[row['page_id']]
        assert row['label']==1 and row['split']==page['split']=='train'
        box=np.asarray(row['bbox_xyxy'],dtype=np.float64)
        gt=np.asarray([a['bbox'] for a in page['annotations']],dtype=np.float64)
        overlap=ious(box[None],gt)[0];gi=int(np.argmax(overlap));target=gt[gi]
        assert overlap[gi]>=.5 and abs(overlap[gi]-row['max_annotation_IoU'])<1e-10
        width,height=box[2:]-box[:2];tw,th=target[2:]-target[:2]
        center=(box[:2]+box[2:])/2;tc=(target[:2]+target[2:])/2
        y=[(tc[0]-center[0])/width,(tc[1]-center[1])/height,np.log(tw/width),np.log(th/height)]
        targets.append(y)
        provenance.append({'source_index':i,'source_id':row['id'],'page_id':row['page_id'],
                           'annotation_index':gi,'proposal':box.tolist(),'target':target.tolist(),
                           'label_basis':row['label_basis'],'split':'train'})
    y=np.asarray(targets,dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    model=Ridge(alpha=10).fit(x,y)
    output.mkdir()
    np.savez_compressed(output/'features_targets.npz',x=x,y=y)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance))
    joblib.dump(model,output/'model.joblib',compress=3)
    report={'split':'train','samples':len(rows),'pages':len({r['page_id'] for r in rows}),
            'alpha':10,'seed':'none; deterministic original source order and nearest GT argmax',
            'input':'324-dimensional grayscale HOG; normalized dx,dy,log width,log height',
            'pool_provenance_sha256':sha(pool/'provenance.json'),'source_audit_sha256':sha(audit),
            'manifest_sha256':sha(manifest),'features_targets_sha256':sha(output/'features_targets.npz'),
            'samples_sha256':sha(output/'samples.jsonl'),'model_sha256':sha(output/'model.joblib'),
            'script_sha256':sha(Path(__file__)),
            'limitation':'Includes repeated-face GT jitter; all source IoU>=.5; effect on lower-overlap false proposals must be evaluated. No validation/test fitting.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
