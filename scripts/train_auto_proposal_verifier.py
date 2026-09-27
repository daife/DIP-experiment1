"""Automatically rebuild HOG training around the new Cascade's train proposals."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.candidate_verifier import hog,features
from src.cascade import cascade_from_dict
from src.multiscale import detect_multiscale,PyramidConfig
from train_candidate_verifier import choose_pages,iou_max


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'datasets/derived/cascade_auto_proposal_verifier_v1'
    output.mkdir(parents=True,exist_ok=False)
    cascade_path=ROOT/'datasets/derived/cascade_auto_curated_localization_v1/model.json'
    manifest=ROOT/'datasets/manifests/normalized/manga109_faces.jsonl'
    cascade=cascade_from_dict(json.loads(cascade_path.read_text()))
    seed=20260926;rng=np.random.default_rng(seed)
    base=ROOT/'datasets/derived/cascade_auto_v1'
    labels=np.load(base/'labels.npy');data=np.load(base/'channels.npy',mmap_mode='r')
    base_rows=list(map(json.loads,(base/'samples.jsonl').read_text().splitlines()))
    # Fixed random train source faces, without selecting by old HOG scores.
    positive_indices=np.sort(rng.choice(np.flatnonzero(labels==1),4000,replace=False))
    x=[hog(data[i,0]) for i in positive_indices];y=[1]*len(x)
    provenance=[{'dataset':'cascade_auto_v1','source_index':int(i),'source_id':base_rows[i]['id'],'label':1} for i in positive_indices]
    assert all(base_rows[i]['split']=='train' for i in positive_indices)
    sources=[]
    for name in ('cascade_auto_background_merged_v1','cascade_localization_negatives_v1'):
        source=ROOT/'datasets/derived'/name
        rows=list(map(json.loads,(source/'candidates.jsonl').read_text().splitlines()))
        cache=np.load(source/'channels11.npy',mmap_mode='r')
        indices=np.sort(rng.choice(len(rows),4000,replace=False))
        for i in indices:
            row=rows[i]
            assert row['split']=='train' and row['label']==0
            assert hashlib.sha256(cache[i,0].tobytes()).hexdigest()==row['crop_sha256']
            x.append(hog(cache[i,0]));y.append(0)
            provenance.append({'dataset':name,'source_index':int(i),'source_id':row['id'],'label':0})
        sources.append({'dataset':name,'manifest_sha256':sha(source/'candidates.jsonl'),'selected':len(indices)})
    pages=[]
    for page in choose_pages(manifest,32):
        assert page['split']=='train'
        path=ROOT/'datasets'/page['image_path'];gray=cv2.imread(str(path),0)
        if gray is None:
            raise ValueError(path)
        boxes,scores,_=detect_multiscale(gray,cascade,PyramidConfig(1.2,2,.3))
        gt=np.asarray([a['bbox'] for a in page['annotations']]).reshape(-1,4)
        overlap=iou_max(boxes,gt)
        indices=np.flatnonzero(overlap>=.5)
        vectors=features(gray,boxes[indices])
        for i,vector in zip(indices,vectors):
            x.append(vector);y.append(1)
            provenance.append({'image_id':page['image_id'],'image_path':page['image_path'],'split':'train',
                'bbox':boxes[i].tolist(),'label':1,'max_annotation_IoU':float(overlap[i])})
        item={'image_id':page['image_id'],'source_image_sha256':sha(path),'GT':len(gt),'proposals':len(boxes),'positive_proposals':len(indices)}
        pages.append(item);print(item,flush=True)
        (output/'progress.json').write_text(json.dumps(pages,indent=2)+'\n')
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.uint8)
    np.savez_compressed(output/'features.npz',x=x,y=y)
    (output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance))
    model=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced',cache_size=1024,random_state=seed)
    model.fit(x,y);joblib.dump(model,output/'model.joblib',compress=3)
    report={'split':'train','seed':seed,'positive':int(y.sum()),'negative':int((y==0).sum()),'pages':pages,
        'sources':sources,'base_manifest_sha256':sha(base/'samples.jsonl'),'cascade_sha256':sha(cascade_path),
        'manifest_sha256':sha(manifest),'features_sha256':sha(output/'features.npz'),'samples_sha256':sha(output/'samples.jsonl'),
        'model_sha256':sha(output/'model.joblib'),'script_sha256':sha(Path(__file__)),
        'C':1,'gamma':'scale','class_weight':'balanced','support_vectors':len(model.support_vectors_),
        'limitation':'Automatically labelled, not human-reviewed; localization negatives may contain partial faces; old source annotation omissions and domain bias remain possible. No validation/test used to fit.'}
    (output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    main()
